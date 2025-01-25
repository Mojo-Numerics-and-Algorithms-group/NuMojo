import json
from mdutils.mdutils import MdUtils
from mdutils import Html
import os
from pathlib import Path


print("Ignore the syntax warnings the behavoir handles Variadic list asignment def func(*variable)")

docs = json.load(open("docs/readthedocs/docs.json"))

def doc_func(func,mdfile:MdUtils,top_header=2):
    mdfile.new_header(top_header,f"""{func["name"]}""")
    for overload in func["overloads"]:
        mdfile.insert_code(f"""{overload["signature"]}""",language="Mojo")
        mdfile.new_line("""Summary""")
        mdfile.new_line()
        mdfile.new_line(overload["summary"])
        mdfile.new_line()
        
        paramlist = list()
        
        if overload["parameters"]:
            mdfile.new_line("""Parameters:""")
            mdfile.new_line()
            for param in overload["parameters"]:
                if "*" not in param["name"]:
                        name =param["name"] 
                else:
                    name = param["name"].replace("*","\*")
                if param["description"]:
                    description = f': {param["description"]}'
                else:
                    description = ""
                if "default" in  list(param.keys()):
                    default = f""" Defualt: `{param["default"]}`"""
                else:
                    default = ""
                paramlist.append(name + description + default)
            mdfile.new_list(paramlist)

        if overload["constraints"]:
            mdfile.new_line("""Constraints:""")
            mdfile.new_paragraph(overload["constraints"])
            mdfile.new_line()

        arglist = list()
        if overload["args"]:
            mdfile.new_line("""Args:""")
            mdfile.new_line()
            for arg in overload["args"]:
                if "*" not in arg["name"]:
                    name =arg["name"] 
                else:
                    name = arg["name"].replace("*","\*")
                if arg["description"]:
                    description = f': {arg["description"]}'
                    
                else:
                    description = ""
                if "default" in  list(arg.keys()):
                    default = f""" Default: {arg["default"]}"""
                else:
                    default = ""
                arglist.append(name + description + default)
            mdfile.new_list(arglist)
            if overload["description"]:
                mdfile.new_paragraph(f"""{overload["description"]}""")


def doc_alias(alias,mdfile:MdUtils,top_header=2):
    if not alias:
        return
    mdfile.new_header(top_header,"Aliases")
    for al in alias:
        mdfile.new_line(f"""`{al["name"]}`: {al["summary"]}""")

def doc_struct(struct, mdfile:MdUtils,top_header=1):
    mdfile.new_header(top_header,struct["name"])
    mdfile.new_header(top_header+1,f"{struct['name']} Summary")
    mdfile.new_line()
    try:
        mdfile.new_line(struct["summary"])
    except:
        print(struct["name"])
    mdfile.new_line()
    if struct['parentTraits']:
        mdfile.new_header(top_header+1,"Parent Traits")
        mdfile.new_line()
        mdfile.new_list(struct['parentTraits'])
    if "aliases" in list(struct.keys()):
        if struct["aliases"]:
            doc_alias(struct["aliases"],mdfile,top_header=top_header+1)
    if struct["fields"]:
        mdfile.new_header(top_header+1,"Fields")
        mdfile.new_line()
        for field in struct["fields"]:
            mdfile.new_line(f"* {field['name']} `{field['type']}`")
            if field["summary"]:
                mdfile.new_line("    - " + field["summary"])
    mdfile.new_line()           
    mdfile.new_header(top_header+1,"Functions")
    if struct["functions"]:
        for func in struct["functions"]:
            doc_func(func, mdfile,top_header=top_header+2)

def doc_modules(module, root:Path, parent:Path):
    mdfile = MdUtils(str(root/parent)+"/"+module["name"])
    mdfile.new_header(1,module["name"])
    mdfile.new_header(2," Module Summary")
    mdfile.new_line(module["summary"])
    if module["aliases"]:
        doc_alias(module["aliases"], mdfile=mdfile)
    if module["structs"]:
        for struct in module["structs"]:
            doc_struct(struct, mdfile,top_header=2)
    if module["traits"]:
        for trait in module["traits"]:
            doc_struct(trait, mdfile,top_header=2)
    if module["functions"]:
        for func in module["functions"]:
            doc_func(func, mdfile,top_header=2)
    mdfile.create_md_file()

def doc_package(package, root, parent):
    
    Path(str(root/parent)+"/"+package["name"]).mkdir(parents=True,exist_ok=True)
    
    if package["packages"]:
        for pack in package["packages"]:
            Path(str(root/parent)+"/"+package["name"]).mkdir(parents=True,exist_ok=True)
            doc_package(pack,root,Path(str(parent)+"/"+package["name"]))
    if package["modules"]:
        for mod in package["modules"]:
            if mod["name"] == "__init__":
                continue
            doc_modules(mod,root,str(parent)+"/"+package["name"])

if __name__ == "__main__":
    for pack in docs["decl"]["packages"]:
        doc_package(pack,Path("./docs/readthedocs/docs/autodocs"),Path(""))

