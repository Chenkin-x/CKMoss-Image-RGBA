"""Packed, variable-length text-to-image forwards using the original MOSS layers."""
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from flash_attn.cute import flash_attn_varlen_func


def cumulative(lengths, device):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)


def attention(q, k, v, cu, maximum):
    out, _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=maximum, max_seqlen_k=maximum, causal=False)
    return out


class MossPackedModel(nn.Module):
    def __init__(self, base_model, gradient_checkpointing=False):
        super().__init__()
        self.base_model = base_model
        self.gradient_checkpointing = gradient_checkpointing

    @property
    def config(self):
        return self.base_model.config

    @staticmethod
    def _rotate(x, rope):
        sin, cos = rope.chunk(2, -1)
        sin = torch.cat((sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin

    def _times(self, times):
        m = self.base_model
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = (times.double() * 1000)[:, None] * freq
        sinusoid = torch.cat((phase.cos(), phase.sin()), -1).float()
        t = m.time_embedding(sinusoid.to(m.time_embedding[0].weight.dtype)).float()
        mod = m.shared_block_modulation(t.to(m.shared_block_modulation.weight.dtype)).view(-1,6,self.config.dim)
        head = m.video_head_modulation(t.to(m.video_head_modulation.weight.dtype)).view(-1,2,self.config.dim)
        return mod, head

    def _text(self, contexts, ropes):
        m = self.base_model.text_embedder
        lengths = [len(x) for x in contexts]
        cu = cumulative(lengths, contexts[0].device)
        rope = torch.cat(ropes)[None]
        x = m.proj(torch.cat(contexts).to(m.proj.weight.dtype))
        for block in m.blocks:
            q,k,v = block.qkv(block.norm1(x)).chunk(3,-1)
            shape = (-1,block.heads,x.shape[-1]//block.heads)
            q = block.q_norm(q.reshape(shape))
            k = block.k_norm(k.reshape(shape))
            q = self._rotate(q[None],rope)[0].to(v.dtype)
            k = self._rotate(k[None],rope)[0].to(v.dtype)
            y = attention(q,k,v.reshape(shape),cu,max(lengths))
            x = x + block.proj(y.flatten(1))
            y = block.norm2(x)
            h,g = block.ffn_up(y),block.ffn_gate(y)
            x = x + block.ffn_down((4*torch.tanh(g/4)*torch.sigmoid(g))*(25*torch.tanh(h/25)))
        return x.split(lengths)

    def _packed_block(self, layer, x, rope, mod, layout):
        # Each image and its text share a timestep.
        ids = layout['time_ids']
        y = layer.norm1(x)*(1+mod[:,1][ids]) + mod[:,0][ids]
        y = y.to(layer.proj['q'].weight.dtype)
        shape = (-1,layer.heads,self.config.dim//layer.heads)
        q = layer.q_norm(layer.proj['q'](y).reshape(shape))
        k = layer.k_norm(layer.proj['k'](y).reshape(shape))
        v = layer.proj['v'](y).reshape(shape)
        q = self._rotate(q[None],rope[None])[0].to(v.dtype)
        k = self._rotate(k[None],rope[None])[0].to(v.dtype)
        attended = attention(q,k,v,layout['cu'],layout['maximum']).flatten(1)
        x = x + layer.proj['o'](attended)*mod[:,2][ids]
        y = layer.norm2(x)*(1+mod[:,4][ids]) + mod[:,3][ids]
        h,g = layer.proj['ffn_up_gate'](y.to(layer.proj['q'].weight.dtype)).chunk(2,-1)
        h = (4*torch.tanh(g/4)*torch.sigmoid(g))*(25*torch.tanh(h/25))
        return x + layer.proj['ffn_down'](h)*mod[:,5][ids]

    def forward(self, images, times, contexts):
        m, device = self.base_model, images[0].device
        body_ropes = [m.rope(x.shape[-2], x.shape[-1], len(c), device)[0].to(x.dtype)
                      for x, c in zip(images, contexts)]
        text_ropes = [r[x.shape[-2] * x.shape[-1]:] for r, x in zip(body_ropes, images)]
        text = self._text(contexts, text_ropes)
        tokens, ropes, target_indices, target_ids, time_ids, lengths = [], [], [], [], [], []
        offset = 0
        for i, (image, context, rope) in enumerate(zip(images, text, body_ropes)):
            video = m.video_embedder(image.flatten(1).T.to(m.video_embedder.weight.dtype))
            nv, nt = len(video), len(context)
            tokens.append(torch.cat((context, video)))
            ropes.append(torch.cat((rope[nv:], rope[:nv])))
            target_indices.extend(range(offset + nt, offset + nt + nv))
            target_ids.extend([i] * nv)
            time_ids.extend([i] * (nt + nv))
            lengths.append(nt + nv)
            offset += nt + nv
        layout = {'cu': cumulative(lengths, device), 'maximum': max(lengths),
                  'time_ids': torch.tensor(time_ids, device=device, dtype=torch.long)}
        mod, head = self._times(times)
        x, rope = torch.cat(tokens), torch.cat(ropes)
        for layer in m.layers:
            fn = partial(self._packed_block, layer, layout=layout)
            x = checkpoint(fn, x, rope, mod, use_reentrant=False) if self.gradient_checkpointing and self.training else fn(x, rope, mod)
        targets = torch.tensor(target_indices, device=device, dtype=torch.long)
        ids = torch.tensor(target_ids, device=device, dtype=torch.long)
        x = m.final_norm_video(x[targets].to(m.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1][ids]) + head[:, 0][ids]
        x = m.final_linear_video(x.to(m.final_linear_video.weight.dtype))
        return [out.T.reshape_as(im) for out, im in zip(x.split([im.shape[-2] * im.shape[-1] for im in images]), images)]
