"""Project-only launcher: keep official checkpoint saves, omit duplicate final root export."""
import runpy,sys
from pathlib import Path
from transformers import Trainer
root=Path(__file__).resolve().parents[1]
original=Trainer.save_model

def save_single(self,output_dir=None,_internal_call=False):
    if output_dir is None and not _internal_call:
        final=Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
        if (final/'model.safetensors.index.json').exists() and list(final.glob('model-*.safetensors')):
            print(f'C single-export: final checkpoint retained at {final}; duplicate root weights omitted.',flush=True)
            self.save_state()
            return
    return original(self,output_dir=output_dir,_internal_call=_internal_call)

Trainer.save_model=save_single
runpy.run_path(str(root/'simulation/Isaac-GR00T/gr00t/experiment/launch_finetune.py'),run_name='__main__')
