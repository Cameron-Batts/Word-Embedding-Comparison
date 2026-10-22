NOTEBOOK='notebooks/word_embedding_comparison.ipynb'
REPORT_DIR='reports'
STOP_AFTER=None
REDACT=True
PACKAGES=['numpy', 'pandas', 'gensim', 'nltk', 'scipy']
SCOPE='Full default workflow: four 10-epoch embedding models. Optional Google News download disabled. Neighbor outputs kept local; multiworker nondeterminism remains.'
def inspect_cell(i,s,r):pass
def summarize(s,r):
 assert len(s['processed_text'])==2000 and not s['RUN_PRETRAINED']
 return {'reviews':2000,'models':{name:{'vocabulary_size':len(s[name].wv),'vector_size':s[name].vector_size,'epochs':s[name].epochs,'vampire_vampires_similarity':float(s[name].wv.similarity('vampire','vampires'))} for name in ['w2v_cbow','w2v_sg','ft_cbow','ft_sg']},'optional_pretrained_run':False}

"""Execute the project workflow and save fresh aggregate validation evidence."""
import ast,base64,contextlib,hashlib,io,json,os,platform,time
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]
os.chdir(root)
sys_path=str(root)
import sys
sys.path.insert(0,sys_path)
path=root/NOTEBOOK
nb=json.loads(path.read_text())
scope={}
records={}
count=0
figures=0
cell_outputs=[]
old_show=plt.show
started=time.time()
def capture(*args,**kwargs):
 global figures
 for number in plt.get_fignums():
  figure=plt.figure(number);buffer=io.BytesIO()
  figure.savefig(buffer,format='png',dpi=90,bbox_inches='tight')
  cell_outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':['<Rerun figure>']}})
  figures+=1;plt.close(figure)
plt.show=capture
try:
 for i,cell in enumerate(nb['cells']):
  if cell['cell_type']!='code' or (STOP_AFTER is not None and i>STOP_AFTER):continue
  print('Running cell',i,flush=True)
  cell_outputs=[];stream=io.StringIO();value=None;count+=1
  tree=ast.parse(''.join(cell['source']))
  with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
   if tree.body and isinstance(tree.body[-1],ast.Expr):
    expression=tree.body.pop().value;exec(compile(tree,f'cell-{i}','exec'),scope)
    value=eval(compile(ast.Expression(expression),f'cell-{i}','eval'),scope)
   else:exec(compile(tree,f'cell-{i}','exec'),scope)
   capture()
  if REDACT:
   text='Executed during validation; source-text and neighbor displays are omitted from saved outputs. See the aggregate validation record.\n'
  else:text=stream.getvalue()
  if text:cell_outputs.insert(0,{'output_type':'stream','name':'stdout','text':text.splitlines(True)})
  if value is not None and not REDACT and not hasattr(value,'fit') and not hasattr(value,'figure'):
   cell_outputs.append({'output_type':'execute_result','execution_count':count,'metadata':{},'data':{'text/plain':[repr(value)]}})
  cell['execution_count']=count;cell['outputs']=cell_outputs
  inspect_cell(i,scope,records)
finally:plt.show=old_show
summary=summarize(scope,records)
versions={}
for package in PACKAGES:
 try:versions[package]=version(package)
 except PackageNotFoundError:pass
inputs={}
for base in ['data','Data']:
 if (root/base).is_dir():
  for f in (root/base).iterdir():
   if f.suffix.lower() in ['.csv','.rds']:inputs[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
summary.update({'checked_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'versions':versions,'executed_code_cells':count,'generated_figures':figures,'elapsed_seconds':time.time()-started,'input_sha256':inputs,'scope':SCOPE})
report=root/REPORT_DIR/'validation_results.json';report.write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
nb['metadata'].setdefault('language_info',{})['version']=platform.python_version()
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,default=str),flush=True)
