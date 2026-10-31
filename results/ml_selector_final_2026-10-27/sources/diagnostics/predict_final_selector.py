"""Working coefficient-to-profile RESEARCH inference; never runs a solver."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.final_selector import FinalResearchSelector


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path)
    args=ap.parse_args();payload=json.loads(args.input.read_text());model=FinalResearchSelector(args.model)
    try:result=model.predict(payload['coefficients'],payload['x_grid'],payload['coefficient_metadata'],domain=payload.get('domain','all_methods_v1'),profile_context=payload.get('profile_context'),timing_context=payload.get('timing_context'))
    except (ValueError,KeyError,TypeError) as exc:result=dict(status='invalid_input',action='abstain',error=str(exc),selected_profile=None,research_only=True)
    text=json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(text)
    else:print(text,end='')
    return 2 if result['status']=='invalid_input' else 0


if __name__=='__main__':raise SystemExit(main())
