import copy,json,random
from pathlib import Path
import torch
from open_instruct.portable import train,model,data
cfg=train.Config(**json.loads(Path('configs/portable/tulu3-real.json').read_text()))
d=data.load('experiments/tulu3_real/data/dataset.json')
torch.set_num_threads(4); torch.manual_seed(1);torch.mps.manual_seed(1);random.seed(1)
lm,tok=model.create(cfg.model,cfg.revision,cfg.max_length);lm.to('mps').eval()
# Reconstruct the initial evaluation's RNG effects exactly before resuming SFT selection.
# Sampling indices are recorded here for a bounded independent reproduction.
enc=[data.encode(tok,r,'chosen',cfg.max_length) for r in d['sft']]
opt=torch.optim.AdamW(lm.parameters(),lr=cfg.learning_rate,weight_decay=0.)
batches=train.sampled_batches(list(range(len(enc))),cfg)
for step in range(1,97):
 ix=next(batches);selected=[enc[i] for i in ix];total=sum(len(x)-p for x,p in selected)
 for micro in train.chunks(selected,cfg.micro_batch_size):
  tokens,attention,mask=model.batch(micro,tok.pad_token_id,'mps');loss=-(model.logps(lm,tokens,attention)*mask).sum()/total;loss.backward()
 bad=[name for name,p in lm.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
 print(step,ix,[len(x) for x,p in selected],loss.item(),bad[:3],flush=True)
 if bad:
  cpu=copy.deepcopy(lm).cpu();cpu.zero_grad()
  for micro in train.chunks(selected,cfg.micro_batch_size):
   tokens,attention,mask=model.batch(micro,tok.pad_token_id,'cpu');loss=-(model.logps(cpu,tokens,attention)*mask).sum()/total;loss.backward()
  cpu_bad=[name for name,p in cpu.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
  Path('experiments/tulu3_real/mps-gradient-diagnosis.json').write_text(json.dumps(dict(step=step,indices=ix,mps_bad=bad,cpu_bad=cpu_bad,cpu_loss=loss.item()),indent=2))
  break
 train.learning_rate(opt,cfg.learning_rate,step,cfg.sft_steps,cfg)
 train.step_optimizer(loss,opt,lm.parameters())
