"""Read-only official GitHub/PyPI version inventory; no repository updates."""
import concurrent.futures, json, subprocess, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
repos={'Isaac-GR00T':('NVIDIA/Isaac-GR00T','simulation/Isaac-GR00T'),'RoboTwin':('RoboTwin-Platform/RoboTwin','simulation/RoboTwin'),'XPolicyLab':('XPolicyLab/XPolicyLab','simulation/RoboTwin/XPolicyLab'),'LeRobot':('huggingface/lerobot',None)}
def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GR00T-recovery-audit'}),timeout=45) as r: return json.load(r)
    except Exception as e: return {'error':str(e)}
def audit(item):
    name,(repo,path)=item; result={'repository':repo,'url':f'https://github.com/{repo}'}
    if path:
        for key,args in [('local_commit',['rev-parse','HEAD']),('local_date',['show','-s','--format=%cI','HEAD']),('local_changes',['status','--short'])]: result[key]=subprocess.check_output(['git','-C',str(root/path),*args],text=True).strip()
    for key,endpoint in [('latest_release','releases/latest'),('tags','tags?per_page=5'),('latest_commit','commits?per_page=1'),('recent_commits','commits?per_page=15')]:
        data=get(f'https://api.github.com/repos/{repo}/{endpoint}')
        if key=='latest_release' and 'error' not in data: data={k:data.get(k) for k in ['tag_name','published_at','target_commitish','html_url','body']}
        if 'commit' in key and isinstance(data,list): data=[{'sha':d['sha'],'date':d['commit']['committer']['date'],'message':d['commit']['message'],'url':d['html_url']} for d in data]
        result[key]=data
    return name,result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=dict(pool.map(audit,repos.items()))
for package in ['sapien','lerobot']: 
    d=get(f'https://pypi.org/pypi/{package}/json'); results[package+'_pypi']={'version':d.get('info',{}).get('version'),'releases':{k:[{'date':x['upload_time_iso_8601'],'requires_python':x['requires_python']} for x in v[:1]] for k,v in d.get('releases',{}).items()}}
(root/'evaluation/framework_upstream_audit.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
