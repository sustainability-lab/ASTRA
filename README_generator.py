import os
import subprocess
from glob import glob
from os.path import join
import jinja2

# Set torch HOME
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")

# Load a template named "README_template.jinja2"
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
TEMPLATE_FILE = "README_template.md"

# Load examples
example_dict = {}
examples_dir = "quick_examples"
files = glob(join(examples_dir, "*.py"))

for i, file in enumerate(files):
    print(f"Running {file}")
    with open(file) as f:
        filename = os.path.basename(file).split(".")[0]
        example_dict[filename] = f.read()
        # execute py `file` and get output
        process = subprocess.run(["python", file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        example_dict[filename + "_output"] = process.stdout.decode("utf-8")
        example_dict[filename + "_error"] = process.stderr.decode("utf-8")


readme = templateEnv.get_template(TEMPLATE_FILE).render(**example_dict)

# Write README.md
with open("README.md", "w") as f:
    f.write(readme)

print("README.md generated")
