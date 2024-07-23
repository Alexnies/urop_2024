import os

os.environ['GPROMSHOME'] = '/usr/local/pse/gPROCESS_2023/gPROMS-core_2022.2.2.55277'

# Add the directory to PYTHONPATH
go_python_path = '/usr/local/pse/gPROCESS_2023/gPROMS-core_2022.2.2.55277/bin'

if go_python_path not in os.getenv("PATH"):
    os.environ["PATH"] += os.pathsep + go_python_path

print("GPROMSHOME:", os.getenv("GPROMSHOME"))
print("PATH:", os.getenv("PATH"))

import gopython

status = gopython.start_only()
status = gopython.select("gOPythonSamanVariableGroupInput","gOPythonSamanVariableGroupInput")
status = gopython.simulate("gOPythonSamanVariableGroupInput")