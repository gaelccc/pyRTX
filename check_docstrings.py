import os
import ast

def check_docstrings(directory):
    missing_docstrings = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=filepath)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                if not ast.get_docstring(node):
                                    missing_docstrings.append(f"{filepath}:{node.lineno}:{node.name}")
                    except SyntaxError as e:
                        print(f"Could not parse {filepath}: {e}")
    return missing_docstrings

if __name__ == "__main__":
    missing = check_docstrings("pyRTX")
    if missing:
        print("Functions or classes missing docstrings:")
        for item in missing:
            print(f"- {item}")
    else:
        print("All functions and classes have docstrings.")
