import git
import os

from pandas import DataFrame
from pathlib import Path

import call_graph.parsers as parsers


def reset_graph():
    global method_dict, file_dict, edge_dict
    method_dict = {
        'method': [],
        'docstring': [],
        'nodes': [],
        'prints': [],
        'path': []
    }
    file_dict = {}
    edge_dict = {
        'callee_index': [],
        'called_index': []
    }


# From: https://github.com/github/CodeSearchNet/tree/master/function_parser
def traverse(
    node,       # tree-sitter node
    results,    # list to append results to
) -> None:
    """Traverse in a recursive way, a tree-sitter node and append results to a list."""
    if node.type == 'string':
        results.append(node)
        return
    for n in node.children:
        traverse(n, results)
    if not node.children:
        results.append(node)


def get_docstring_method_pairs(root_node):
    """Get all docstrings and methods from a tree-sitter root node."""
    docstring_method_pairs = []
    query = lang.docstring_method_import_q
    captures = query.captures(root_node)
    for i, (node, type_id) in enumerate(captures):
        if type_id == "method":
            previous_node, previous_type_id = None, None
            if i > 0:
                previous_node, previous_type_id = captures[i-1]
                if previous_node.end_point[0] != node.start_point[0] - 1:
                    previous_node, previous_type_id = None, None
                elif previous_type_id != "docstring":
                    previous_node, previous_type_id = None, None
            docstring_method_pairs.append([previous_node, node])
    return docstring_method_pairs


def add_methods_and_imports(dir_path, include_docstrings=False):
    tree = lang.PARSER.parse(bytes(lang.src_code, "utf8"))
    query = lang.method_import_q
    captures = query.captures(tree.root_node)
    if include_docstrings:
        docstring_method_pairs = get_docstring_method_pairs(tree.root_node)
        cur_method_nodes = [method for _, method in docstring_method_pairs]
        curr_docstring_nodes = [
            docstring for docstring, _ in docstring_method_pairs
        ]
    else:
        query = lang.method_import_q
        captures = query.captures(tree.root_node)
        # adds all the method nodes to a list and all the method definition to a dictionary
        cur_method_nodes = [node[0] for node in captures if node[1] == 'method']
        curr_docstring_nodes = [None for node in captures if node[1] == 'method']

    method_dict['method'].extend([lang.node_to_string(node) for node in cur_method_nodes])
    method_dict['docstring'].extend(
        [
            lang.node_to_string(node)
            if node is not None else ""
            for node in curr_docstring_nodes
        ]
    )
    method_dict['path'].extend([lang.filepath for node in cur_method_nodes])
    method_dict['nodes'].extend(cur_method_nodes)
    method_dict['prints'].extend([lang.get_method_print(node) for node in cur_method_nodes])
    ## adds all files that the file imports to a list and the range of indexes in the method dictionary that point to that file
    import_nodes = [node[0] for node in captures if node[1] == 'import']

    file_list =[]
    parent_dir = lang.filepath[:lang.filepath.rfind('/')]
    for f in os.listdir(parent_dir):
        if f.endswith(lang.extension):
            file_list.append(os.path.join(parent_dir, f))

    for imp in import_nodes:
        file_to_search = lang.get_import_file(imp)
        if file_to_search.startswith(lang.language):
            continue
        try:
            filepath_parts = lang.filepath.split('/')
            file_to_search_start = file_to_search.split('/')[0]
            if file_to_search_start in filepath_parts:
                import_path_start = len(filepath_parts) - filepath_parts[::-1].index(file_to_search_start) - 1
                import_path = '/'.join(filepath_parts[:import_path_start]) + '/' + file_to_search
                if os.path.exists(import_path):
                    file_list.append(import_path)
                    continue

            paths = Path(dir_path).rglob(file_to_search)
            for path in paths:
                if '/test/' in str(path) + '/':
                    continue
                if path.exists():
                    file_list.append(str(path))
                    break
        except Exception as e:
            pass

    file_dict[lang.filepath] = [file_list, (len(method_dict['nodes']) - len(cur_method_nodes), len(method_dict['nodes'])), tree.root_node]


def add_edges():
    query = lang.call_q
    file_path = lang.filepath
    method_range = file_dict[file_path][1]
    imports = file_dict[file_path][0]
    for index in range(method_range[0], method_range[1]):
        node = method_dict['nodes'][index]
        calls = [call[0] for call in query.captures(node)]
        for call in calls:
            call_name = lang.get_call_print(call)
            if not call_name[0]:
                continue
            for file in imports:
                # locate the target file containing the callee method
                if file.split('/')[-1][:-5] == call_name[0]:
                    rang = file_dict[file][1]
                    flag = 0
                    for jindex in range(rang[0], rang[1]):
                        method_name = method_dict['prints'][jindex]
                        # find the callee method in the current file
                        if call_name[1:] == method_name:
                            edge_dict['callee_index'].append(index)
                            edge_dict['called_index'].append(jindex)
                            flag = 1
                    # futher search for the callee method in the superclass
                    if not flag:
                        superclass = ''
                        root_node = file_dict[file][2]
                        # obtain the name of the superclass
                        for child in root_node.children:
                            if child.type == 'class_declaration':
                                if child.child_by_field_name('superclass'):
                                    lang.set_current_file(file)
                                    superclass = lang.node_to_string(child.child_by_field_name('superclass'))[8:]
                                    lang.set_current_file(file_path)
                                break
                        if superclass:
                            for imp in file_dict[file][0]:
                                if imp.split('/')[-1][:-5] == superclass:
                                    r = file_dict[imp][1]
                                    for j in range(r[0], r[1]):
                                        mtd_name = method_dict['prints'][j]
                                        if call_name[1:] == mtd_name:
                                            edge_dict['callee_index'].append(index)
                                            edge_dict['called_index'].append(j)
                                    break
                    break


def set_language(language):
    global lang
    if language == 'python':
        lang = parsers.PythonParser()
    elif language == 'java':
        lang = parsers.JavaParser()
    elif language == 'cpp':
        lang = parsers.CppParser()


def parse_directory(dir_path, include_docstring=False) -> DataFrame:
    reset_graph()
    try:
        if lang is None:
            pass
    except NameError:
        exit_with_message("No language specified")
    if not os.path.isdir(dir_path):
        exit_with_message(f'Could not find directory: {dir_path}')

    dir_path = Path(dir_path)
    paths = dir_path.rglob(f'*{lang.extension}')
    for path in paths:
        if '/test/' in str(path) + '/':
            continue
        lang.set_current_file(str(path))
        add_methods_and_imports(str(dir_path), include_docstring)
    for path in file_dict:
        lang.set_current_file(str(path))
        add_edges()

    return (
        DataFrame(
            {
                'method': method_dict['method'],
                'docstring': method_dict['docstring'],
                'path': method_dict['path']
            }
        ),
        DataFrame(edge_dict)
    )


def exit_with_message(message):
    print(f"{message} Exiting...")
    exit(1)