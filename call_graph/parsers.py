import call_graph
import os

from abc import ABC, abstractmethod
from tree_sitter import Language, Parser


class CallParser():
    __metaclass__ = ABC
    src_code = ''   #A string containing all the source code of the filepath
    lines = []      #All the lines in the current file
    filepath = ''  #the path to the current file

    """A string holding the name of the language, ex: 'python' """
    @property
    @abstractmethod
    def language(self):
        pass

    """A string holding the file extension for the language, ex: '.java' """
    @property
    @abstractmethod
    def extension(self):
        pass

    """A tree-sitter Language object, build from build/my-languages.so """
    @property
    @abstractmethod
    def language_library(self):
        pass

    """A tree-sitter Parser object"""
    @property
    @abstractmethod
    def PARSER(self):
        pass

    """The query that finds the method definitions (including constructors) and import statements"""
    @property
    @abstractmethod
    def method_import_q(self):
        pass

    """The query that finds all the function calls in the file"""
    @property
    @abstractmethod
    def call_q(self):
        pass

    """Sets the current file and updates the src_code and lines"""
    def set_current_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors = 'ignore') as file:
                self.src_code = file.read()
                self.lines = self.src_code.split('\n')
                self.filepath = path
        except FileNotFoundError as err:
            print(err)

    """Takes in a tree-sitter node object and returns the code that it refers to"""
    def node_to_string(self, node) -> str:
        start_point = node.start_point
        end_point = node.end_point
        if start_point[0] == end_point[0]:
            return self.lines[start_point[0]][start_point[1]:end_point[1]]
        ret = self.lines[start_point[0]][start_point[1]:] + "\n"
        ret += "\n".join([line for line in self.lines[start_point[0] + 1:end_point[0]]])
        ret += "\n" + self.lines[end_point[0]][:end_point[1]]
        return ret

    """Takes in a call node and returns a tuple of the name of the method that was called and the number of arguments passed
    for example, if passed the call node 'add(3, 4)' the function will return '(add, 2)'. See the Python, Java, and Cpp parsers
    for example implementations and use https://tree-sitter.github.io/tree-sitter/playground to view the structre of
    a call node in the desired language"""
    @abstractmethod
    def get_call_print(self, call) -> tuple:
        pass

    """Takes in a method node and returns a tuple of the name of the method and the number of parameters passed for example,
    if passed the method node refering to 'def add(a,b)' the function will return '(add, 2)' see Java, Python, and Cpp parsers
    for an example implementation, and use https://tree-sitter.github.io/tree-sitter/playground to view the structre of
    a method call in the desired language"""
    @abstractmethod
    def get_method_print(self, method) -> tuple:
        pass

    """Takes in an import node and returns the path of the file that is imported
    don't wory about filtering out system libraries, as the program will check if the file exitsts before trying to add
    it to the project. You may need to override this method depending on how the language handles imports"""
    def get_import_file(self, imp):
        file_to_search = self.node_to_string(imp)
        return file_to_search.replace(".", os.sep) + self.extension


class CppParser(CallParser):
    pass


class JavaParser(CallParser):
    language = 'java'
    extension = '.java'
    language_library = Language('call_graph/my-languages.so', 'java')
    PARSER = Parser()
    PARSER.set_language(language_library)
    method_import_q = language_library.query("""
            (method_declaration) @method
            (constructor_declaration) @method
            (import_declaration
                (identifier) @import)
            (import_declaration
                (scoped_identifier) @import)
            """)
    docstring_method_import_q = language_library.query("""
            (method_declaration) @method
            (constructor_declaration) @method
            (block_comment) @docstring
            (line_comment) @docstring
            (import_declaration
                (identifier) @import)
            (import_declaration
                (scoped_identifier) @import)
            """)

    call_q = language_library.query("""
            (method_invocation) @call
            """)

    method_in_q = language_library.query("""
            (local_variable_declaration) @lv
            (formal_parameter) @param
            (object_creation_expression) @new
            """)


    field_q = language_library.query("""
            (field_declaration) @field
            """)

    def get_call_print(self, node):
        # gets the name of the method call
        try:
            object_name, class_name = None, None
            class_node, class_new_node = None, None
            object_node = node.child_by_field_name('object')
            method_name = self.node_to_string(node.child_by_field_name('name'))
            nargs = (len(node.child_by_field_name('arguments').children) - 1) // 2

            if object_node:
                if object_node.type == 'identifier':
                    object_name = self.node_to_string(object_node)
                    if object_name[0].isupper():
                        return (object_name, method_name, nargs)
                    else:
                        cur_node = node.parent
                        if cur_node:
                            while cur_node.parent.type != 'class_body' and cur_node.parent.type != 'enum_body' :
                                if cur_node.parent.type == 'interface_body' and cur_node.parent.parent.parent.type == 'program':
                                    break
                                cur_node = cur_node.parent

                        if cur_node.type == 'method_declaration':
                            for item_node in self.method_in_q.captures(cur_node):
                                if item_node[0].start_point[0] < node.start_point[0]:
                                    search_obj_node = None
                                    if item_node[1] == 'param':
                                        search_obj_node = item_node[0].child_by_field_name('name')
                                    elif item_node[1] == 'lv':
                                        search_obj_node = item_node[0].child_by_field_name('declarator').child_by_field_name('name')
                                    else:
                                        if item_node[0].prev_named_sibling and item_node[0].prev_named_sibling.type == 'identifier':
                                            if object_name == self.node_to_string(item_node[0].prev_named_sibling):
                                                class_new_node = item_node[0].child_by_field_name('type')
                                    if search_obj_node and object_name == self.node_to_string(search_obj_node):
                                        class_node = item_node[0].child_by_field_name('type')

                            if not class_new_node and not class_node:
                                cur_node = cur_node.parent
                                fields = [field[0] for field in self.field_q.captures(cur_node) if field[0].start_point[0] < node.start_point[0]]
                                for field in fields:
                                    field_node = None
                                    field_node = field.child_by_field_name('declarator').child_by_field_name('name')
                                    if field_node and self.node_to_string(field_node) == object_name:
                                        class_node = field.child_by_field_name('type')
                                        obj_create_node = None
                                        obj_create_node = field.child_by_field_name('declarator').child_by_field_name('value')
                                        if obj_create_node and obj_create_node.type == 'object_creation_expression':
                                            class_new_node = obj_create_node.child_by_field_name('type')

                elif object_node.type == 'object_creation_expression':
                    class_new_node=object_node.child_by_field_name('type')

                if class_new_node and class_new_node.type == 'type_identifier':
                    class_name = self.node_to_string(class_new_node)
                elif class_node and class_node.type == 'type_identifier':
                    class_name = self.node_to_string(class_node)

            else:
                cur_node = node.parent
                if cur_node:
                    while cur_node.type != 'program':
                        if cur_node.type == 'class_declaration':
                            class_name = self.node_to_string(cur_node.child_by_field_name('name'))
                            break
                        cur_node = cur_node.parent

        except Exception as e:
            print(e)
        return (class_name, method_name, nargs)

    def get_method_print(self, method):
        name = self.node_to_string(method.child_by_field_name('name'))
        nparams = (len(method.child_by_field_name('parameters').children) - 1) // 2

        return (name, nparams)


class PythonParser(CallParser):
    pass