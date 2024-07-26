import networkx as nx
import re
import json
import random

zeroshot_template_dict = {
    'connectivity': ' You should respond in the following format: The answer is {yes/no}.\nA:',
    'shortest_path': ' You should respond in the following format: The shortest path is {shortest path, which is node list seperated by "->"}. The length of shortest path is {shortest path length}.\nA:',
    'topology': ' You should respond in the following format: One possible topological sort is: {topological sort result, which is node number seperated by "->"}.\nA:',
    'flow': ' You should respond in the following format: The maximum flow is {answer}.\nA:'
}

class GraphProblem:
    def __init__(self, format_choice, problem : dict, seed = 0) -> None:
        # general init
        self.node_number, self.graph_str, self.question, self.answer = problem['node_number'], problem['graph'], problem['question'], problem['answer']
        self.format = format_choice
        random.seed(seed)
        self._shuffle_name()

    def _number2letter(self, number):
        """
        Converts a node number (str) to its corresponding letter representation.
        For numbers less than 27, it converts them to a single letter (1 -> A, 26 -> Z).
        For numbers 27 and above, it converts them to a double letter (27 -> AA, 28 -> AB, etc.),
        assuming all node numbers are less than 200.
        """
        number = int(number)
        if number < 26:
            return chr(65 + number)
        else:
            first_letter = chr(64 + (number // 26))
            second_letter = chr(65 + (number % 26))
            return first_letter + second_letter
    
    def _shuffle_name(self):
        names = [
            "Aaron", "Abigail", "Adam", "Adrian", "Aidan", "Aimee", "Alan", "Albert", "Alex", "Alexander",
            "Alice", "Alicia", "Amanda", "Amber", "Amy", "Andrea", "Andrew", "Angela", "Ann", "Anna",
            "Anthony", "Arthur", "Ashley", "Audrey", "Austin", "Barbara", "Benjamin", "Beth", "Betty",
            "Beverly", "Blake", "Bobby", "Bradley", "Brandon", "Brenda", "Brian", "Brittany", "Bruce",
            "Bryan", "Caleb", "Cameron", "Carl", "Carla", "Carol", "Caroline", "Catherine", "Charles",
            "Charlotte", "Cheryl", "Chloe", "Chris", "Christian", "Christina", "Christopher", "Claire",
            "Clara", "Cody", "Colin", "Connor", "Courtney", "Craig", "Crystal", "Cynthia", "Daisy",
            "Daniel", "Danielle", "David", "Dean", "Deborah", "Dennis", "Derek", "Diana", "Diane",
            "Dominic", "Donald", "Donna", "Doris", "Dorothy", "Douglas", "Dylan", "Edward", "Eleanor",
            "Elizabeth", "Ella", "Ellen", "Emily", "Emma", "Eric", "Erin", "Ethan", "Eugene", "Eva",
            "Evan", "Evelyn", "Faith", "Fiona", "Frances", "Frank", "Gabriel", "Gail", "Gary", "George",
            "Gerald", "Gillian", "Grace", "Gregory", "Hannah", "Harold", "Harry", "Hayley", "Heather",
            "Helen", "Henry", "Holly", "Howard", "Ian", "Isaac", "Isabel", "Jack", "Jacob", "Jade",
            "James", "Jane", "Janet", "Jason", "Jean", "Jeffrey", "Jennifer", "Jeremy", "Jessica",
            "Jill", "Joan", "Joanne", "Jodie", "Joe", "John", "Jonathan", "Jordan", "Joseph", "Joshua",
            "Joyce", "Judy", "Julia", "Julie", "Justin", "Karen", "Katherine", "Kathleen", "Keith",
            "Kelly", "Kenneth", "Kevin", "Kimberly", "Kyle", "Laura", "Lauren", "Lawrence", "Leah",
            "Leonard", "Leslie", "Liam", "Lillian", "Lily", "Linda", "Lisa", "Logan", "Lois", "Louise",
            "Lucas", "Lucy", "Lynn", "Madison", "Margaret", "Maria", "Marie", "Marilyn", "Mark",
            "Martha", "Martin", "Mary", "Mason", "Matthew", "Megan", "Melanie", "Melissa", "Michael",
            "Michelle", "Molly", "Nancy", "Natalie", "Nathan", "Nicholas", "Nicole", "Noah", "Nora",
            "Norman", "Oliver", "Olivia", "Oscar", "Pamela", "Patricia", "Patrick", "Paul", "Paula",
            "Peter", "Philip", "Phillip", "Phoebe", "Rachel", "Ralph", "Raymond", "Rebecca", "Richard",
            "Rita", "Robert", "Robin", "Roger", "Ronald", "Rosalind", "Rose", "Ruby", "Russell",
            "Ruth", "Ryan", "Samantha", "Samuel", "Sandra", "Sara", "Sarah", "Scott", "Sean", "Sebastian",
            "Sharon", "Sheila", "Shirley", "Simon", "Sophia", "Stephanie", "Stephen", "Steve", "Steven",
            "Susan", "Sylvia", "Tanya", "Teresa", "Terry", "Thomas", "Timothy", "Tina", "Todd", "Tom",
            "Tracy", "Trevor", "Tyler", "Valerie", "Vanessa", "Veronica", "Victor", "Victoria", "Vincent",
            "Virginia", "Walter", "Wanda", "Wayne", "Wendy", "Wesley", "William", "Willow", "Winifred", "Xander", "Xavier", "Yasmine", "Yvonne", "Zachary", "Zane", "Zara", "Zoe"
        ]
        random.shuffle(names)
        self.names = names
        return

    def _get_name(self, idx):
        assert int(idx) < len(self.names)
        return self.names[int(idx)]
    
    def _generate_format(self):
        if self.format == 'original' or self.format == 'adjacency':
            self.graph_description = self.str2adjacency()
        elif self.format == 'incident':
            self.graph_description = self.str2incident()
        elif self.format == 'graph-expert':
            self.graph_description = self.str2graph_expert()
        elif self.format == 'friendship':
            self.graph_description = self.str2people()
        self.graph = self.str2nxgraph()

    def str2adjacency(self):
        raise NotImplementedError

    def str2incident(self):
        raise NotImplementedError
    
    def str2graph_expert(self):
        raise NotImplementedError
    
    def str2people(self, relation_str='are friends'):
        raise NotImplementedError
    
    def str2nxgraph(self):
        raise NotImplementedError
    
    def get_solution(self):
        raise NotImplementedError
    
    def check_solution(self):
        raise NotImplementedError


class ConnectivityProblem(GraphProblem):
    def __init__(self, format_choice, problem : dict) -> None:
        super().__init__(format_choice, problem)
        
        self.edges = [tuple(map(str, item.strip("()").split(","))) for item in self.graph_str.split(") (")]

        # connectivity init
        self.src, self.trg = [str(i) for i in self.question.split(' ')]

        # format generation
        self._generate_format()

        # get reasoning step
        self.reasoning = self.get_solution()
        self.output_prompt, self.output_answer = self.build_prompt()

    def build_prompt(self):
        '''
        need graph_description, reasoning and answer
        '''
        prompt = ''
        if self.format == 'original' or self.format == 'adjacency':
            prompt += 'Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.\n'
        elif self.format == 'incident':
            prompt += 'The following text describes an undirected graph. Determine if there is a path between two nodes in the graph.\n'
        elif self.format == 'graph-expert':
            prompt += 'You are a graph analyst and you have been given a graph G. G has the following undirected edges:\n'
        elif self.format == 'friendship':
            prompt += 'G describes a friendship graph among the following people. We have the following edges in G:\n'
        prompt += self.graph_description + '\n'
        prompt += f'Q: Is there a path between node {self.src} and node {self.trg}?\nA:'
        answer = f'{self.reasoning}The answer is {self.answer}.'
        return prompt, answer

    def str2adjacency(self):
        """
        Convert an undirected graph (given string representation) to adjacency.
        Input: 
            input_str: A string representing the edges of an undirected graph, e.g., "(1,2) (2,3)"
        Output:
            exactly the same as input
        """
        return self.graph_str

    def str2incident(self):
        """
        Convert an undirected graph (given string representation) to incident.
        Input:
            input_str: A string representing the edges of an undirected graph, 
            e.g., "(1,2) (2,3)" indicates there is an edge between node 1 and 2, and between node 2 and 3.
        Output:
            A string describing each node and its connected neighbors in ascending order,
            e.g., 'Node 1 is connected to 2.\nNode 2 is connected to 1, 3.\nNode 3 is connected to 2.'
        """
        adjacency_list = {}
        for edge in self.edges:
            if edge[0] not in adjacency_list:
                adjacency_list[edge[0]] = []
            adjacency_list[edge[0]].append(edge[1])
            
            if edge[1] not in adjacency_list:
                adjacency_list[edge[1]] = []
            adjacency_list[edge[1]].append(edge[0])

        for key in adjacency_list.keys():
            adjacency_list[key].sort()

        sorted_adjacency_list = dict(sorted(adjacency_list.items()))
        description = []
        for node, neighbors in sorted_adjacency_list.items():
            description.append(f"Node {node} is connected to {', '.join(map(str, neighbors))}.")
        return "\n".join(description)

    def str2graph_expert(self):
        """
        Converts an input string representing graph edges into a string format with letters representing nodes.
        Each edge is represented by 'A -> B' format, where A and B are the letter representations of the nodes.
        
        Input:
            input_str: A string representing the edges of an undirected graph in the format '(1,2) (2,3)'
        Output:
            A string with edges represented in 'A -> B, B -> C' format.
        """
        formatted_edges = [self._number2letter(edge[0]) + " -> " + self._number2letter(edge[1]) for edge in self.edges]
        self.graph_str = ' '.join([f'({self._number2letter(edge[0])},{self._number2letter(edge[1])})' for edge in self.edges])
        self.edges = [tuple(map(str, item.strip("()").split(","))) for item in self.graph_str.split(") (")]
        self.src, self.trg = self._number2letter(self.src), self._number2letter(self.trg)
        return ", ".join(formatted_edges)
    
    def str2people(self, relation_str='are friends'):
        formatted_edges = [f'{self._get_name(edge[0])} and {self._get_name(edge[1])} {relation_str}.' for edge in self.edges]
        self.graph_str = ' '.join([f'({self._get_name(edge[0])},{self._get_name(edge[1])})' for edge in self.edges])
        # print(self.graph_str)
        self.edges = [tuple(map(str, item.strip("()").split(","))) for item in self.graph_str.split(") (")]
        self.src, self.trg = self._get_name(self.src), self._get_name(self.trg)
        return "\n".join(formatted_edges)

    def str2nxgraph(self):
        G = nx.Graph()
        G.add_edges_from(self.edges)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
        # plt.show()
        return G

    def get_solution(self):
        if self.src not in list(self.graph) or self.trg not in list(self.graph):
            return ''
        if not nx.has_path(self.graph, self.src, self.trg):
            return ''
        for path in nx.all_shortest_paths(self.graph, self.src, self.trg):
            # print(path, type(path))
            return ' '.join([f'{path[i]} is connected to {path[i+1]}.' for i in range(len(path)-1)]) + ' '

    def check_solution(self, response):
        return super().check_solution()

class ShortestPathProblem(GraphProblem):
    def __init__(self, format_choice, problem: dict, seed=0, number_convert=int) -> None:
        super().__init__(format_choice, problem, seed)
        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")]
        self.number_convert = number_convert
        # connectivity init
        self.src, self.trg = [str(i) for i in self.question.split(' ')]

        # format generation
        self._generate_format()

        # get reasoning step
        self.reasoning = self.get_solution()
        self.output_prompt, self.output_answer = self.build_prompt()

    def build_prompt(self):
        '''
        need graph_description, reasoning and answer
        '''
        prompt = ''
        if self.format == 'original' or self.format == 'adjacency':
            prompt += 'The following paragraph describes the edges in an undirected graph with weights.\n'
        elif self.format == 'incident':
            prompt += 'The following paragraph describes an undirected graph with weights.\n'
        elif self.format == 'graph-expert':
            prompt += 'You are a graph analyst and you have been given a graph G. G has the following weighted undirected edges:\n'
        elif self.format == 'friendship':
            prompt += 'G describes a friendship graph with distance as weight among the following people. We have the following weighted edges in G:\n'
        prompt += self.graph_description + '\n'
        prompt += f'Q: What is the shortest path between node {self.src} and node {self.trg}, and what is the length of the shortest path?\nA:'
        answer = f'{self.reasoning}The answer is {self.answer}.'
        return prompt, answer

    def str2adjacency(self):
        """
        Convert an undirected graph (given list of tuples) to adjacency.
        Input: 
            A list of tuple representing the weighted edges of an undirected graph, e.g., [(1,2,3),(2, 3,4)]
        Output:
            an edge between node 1 and node 2 with weight 3,
            an edge between node 2 and node 3 with weight 4.
        """
        description = ', '.join([f'an edge between node {edge[0]} and node {edge[1]} has weight {edge[2]}' for edge in self.edges])
        return description + '.'

    def str2incident(self):
        """
        Convert an undirected graph (given list of tuples) to incident.
        Input: 
            A list of tuple representing the weighted edges of an undirected graph, e.g., [(1,2,3),(1,3,4)]
        Output:
            Node 1 is connected to node 2 with weight 3, node 3 with weight 4.
            
        """
        adjacency_list = {}
        for edge in self.edges:
            if edge[0] not in adjacency_list:
                adjacency_list[edge[0]] = []
            adjacency_list[edge[0]].append((edge[1], edge[2]))
            
            if edge[1] not in adjacency_list:
                adjacency_list[edge[1]] = []
            adjacency_list[edge[1]].append((edge[0], edge[2]))

        for key in adjacency_list.keys():
            adjacency_list[key].sort()

        sorted_adjacency_list = dict(sorted(adjacency_list.items()))
        description = []
        for node, neighbors in sorted_adjacency_list.items():
            description.append(f"Node {node} is connected to {', '.join([f'node {neighbors[i][0]} with weight {neighbors[i][1]}' for i in range(len(neighbors))])}.")
        return "\n".join(description)

    def str2graph_expert(self):
        """
        Converts an input string representing graph edges into a string format with letters representing nodes.
        Each edge is represented by 'A -> B' format, where A and B are the letter representations of the nodes.
        
        Input:
            input_str: A string representing the edges of an undirected graph in the format '(1,2) (2,3)'
        Output:
            A string with edges represented in 'A -> B, B -> C' format.
        """
        formatted_edges = [f'{self._number2letter(edge[0])} -> {self._number2letter(edge[1])} with weight {edge[2]}' for edge in self.edges]
        self.graph_str = ','.join([f'{self._number2letter(edge[0])} {self._number2letter(edge[1])} {edge[2]}' for edge in self.edges])
        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")]
        self.src, self.trg = self._number2letter(self.src), self._number2letter(self.trg)
        return ", ".join(formatted_edges)
    
    def str2people(self, relation_str='are friends'):
        formatted_edges = [f'{self._get_name(edge[0])} and {self._get_name(edge[1])} {relation_str}, and they live {edge[2]} miles apart.' for edge in self.edges]
        self.graph_str = ','.join([f'{self._get_name(edge[0])} {self._get_name(edge[1])} {edge[2]}' for edge in self.edges])
        # print(self.graph_str)
        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")]
        self.src, self.trg = self._get_name(self.src), self._get_name(self.trg)
        return "\n".join(formatted_edges)

    def str2nxgraph(self):
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge[0], edge[1], weight=self.number_convert(edge[2]))
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
        # plt.show()
        return G

    def get_solution(self):
        first_sentence = f'Some possible paths from node {self.src} to node {self.trg} are:'
        solution = []
        answer = nx.shortest_path_length(self.graph, self.src, self.trg, weight='weight')
        # for path in nx.all_simple_paths(self.graph, self.src, self.trg, cutoff=answer):
        path = nx.shortest_path(self.graph, self.src, self.trg, weight='weight')
            # print(path, type(path))
        path_weight = [self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])]
        
        path_len = round(sum(path_weight), 1)
        
        solution.append(f"{' -> '.join(path)} with a total weight of {' + '.join([str(weight) for weight in path_weight])} = {path_len}.")

        for i, path in enumerate(nx.all_simple_paths(self.graph, self.src, self.trg, cutoff=len(path))):
            if i > 3:
                break
            path_weight = [self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])]
            path_len = round(sum(path_weight), 1)
            if path_len == answer:
                continue
            solution.append(f"{' -> '.join(path)} with a total weight of {' + '.join([str(weight) for weight in path_weight])} = {path_len}.")
            
        random.shuffle(solution)
        solution = [first_sentence] + solution
        return '\n'.join(solution) + '\n'

    def check_solution(self, response):
        return super().check_solution()


class TopologyProblem(GraphProblem):
    def __init__(self, format_choice, problem: dict) -> None:
        super().__init__(format_choice, problem)

        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")] if self.graph_str else []
        # print(self.edges)

        self._generate_format()

        # get reasoning step
        self.reasoning = self.get_solution()
        self.output_prompt, self.output_answer = self.build_prompt()

    def build_prompt(self):
        '''
        need graph_description, reasoning and answer
        '''
        prompt = ''
        if self.format == 'original' or self.format == 'adjacency':
            prompt += f'In a directed graph with {self.node_number} from 0 to {self.node_number-1}:\n'
        elif self.format == 'incident':
            prompt += f'The following text describes a directed graph among {", ".join([str(i) for i in range(self.node_number)])}.\n'
        elif self.format == 'graph-expert':
            prompt += 'You are a graph analyst and you have been given a graph G. G has the following directed edges:\n'
        elif self.format == 'co-authorship':
            prompt += 'G describes a co-authorship graph among The following people. In this co-authorship graph:\n'
        elif self.format == 'friendship':
            prompt += 'G describes a friendship graph among the following people. We have the following edges in G:\n'
        prompt += self.graph_description + '\n'
        prompt += f'Q: Can all the nodes be visited? Give the solution.\nA:'
        answer = f'{self.reasoning}The answer is:  {self.answer}.'
        return prompt, answer

    def str2adjacency(self):
        pairs = self.graph_str.split(',')
        description = ''
        for pair in pairs:
            a, b = pair.split(' ')
            description += f'Node {a} should be visited before node {b}. '
        return description

    def str2incident(self):
        adjacency_list = {}
        if len(self.edges) == 0:
            return 'There is no edge in this graph.'
        for edge in self.edges:
            if edge[0] not in adjacency_list:
                adjacency_list[edge[0]] = []
            adjacency_list[edge[0]].append(edge[1])
        for key in adjacency_list.keys():
            adjacency_list[key].sort()
        sorted_adjacency_list = dict(sorted(adjacency_list.items()))
        description = []
        for node, neighbors in sorted_adjacency_list.items():
            description.append(f"Node {node} should be visited before nodes {', '.join(map(str, neighbors))}.")
        return "\n".join(description)

    def str2graph_expert(self):
        formatted_edges = [self._number2letter(edge[0]) + " -> " + self._number2letter(edge[1]) for edge in self.edges]
        self.graph_str = ','.join([f'{self._number2letter(edge[0])} {self._number2letter(edge[1])}' for edge in self.edges])
        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")]
        return ", ".join(formatted_edges)

    def str2people(self, relation_str=None):
        return super().str2people(relation_str)

    def str2nxgraph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.node_number))
        for s in self.edges:
            a, b = s
            graph.add_edge(a, b)
        self.graph = graph
        return graph
    
    def get_solution(self):
        paths = self.answer.split(',')
        solution = ''
        for node in paths:
            solution += f"{node} has zero in-degree, we can add it to the answer and substract one from all its neighbors' in-degree. \n"
        return solution.strip('\n')

    def check_solution(self, response):
        return super().check_solution()

class FlowProblem(GraphProblem):
    def __init__(self, format_choice, problem : dict, number_convert=int) -> None:
        super().__init__(format_choice, problem)
        
        self.edges = [tuple(map(str, item.split(" "))) for item in self.graph_str.split(",")]
        # print(self.edges)
        # print(self.question)
        self.src, self.trg = [str(i) for i in self.question.split(' ')]
        self.number_convert = number_convert
        # format generation
        self._generate_format()

        # get reasoning step
        self.reasoning = self.get_solution()
        self.output_prompt, self.output_answer = self.build_prompt()

    def build_prompt(self):
        '''
        need graph_description, reasoning and answer
        '''
        prompt = ''
        if self.format == 'original' or self.format == 'adjacency':
            prompt += f'In a directed graph, the nodes are numbered from 0 to {self.node_number-1}, and the edges are:\n'
        elif self.format == 'incident':
            prompt += f'In a directed graph, the nodes are numbered from 0 to {self.node_number-1}, and the edges are:\n'
        elif self.format == 'graph-expert':
            prompt += 'You are a graph analyst and you have been given a graph G. G has the following directed edges:\n'
        elif self.format == 'co-authorship':
            prompt += 'G describes a co-authorship graph among The following people. In this co-authorship graph:\n'
            # TODO
        elif self.format == 'friendship':
            prompt += 'G describes a friendship graph among the following people. We have the following edges in G:\n'
            # TODO
        prompt += self.graph_description + '\n'
        prompt += f'Q: What is the maximum flow from node {self.src} to node {self.trg}?\nA:'
        answer = f'{self.reasoning}The answer is {self.answer}.'
        return prompt, answer

    def str2adjacency(self):
        """
        Convert an undirected graph (given string representation) to adjacency.
        Input: 
            input_str: A string representing the edges of a directed graph with capacity, e.g., "0 3 5,1 0 3"
        Output:
            output_str: A string representing the edges in texts, e.g., "an edge from node 0 to node 1 with capacity 3, an edge from node 0 to node 2 with capacity 3"
        """
        descriptions = []
        for s in self.graph_str.split(','):
            a, b, cap = s.split(' ')
            descriptions.append(f'an edge from node {a} to node {b} with capacity {cap}')
        return ', '.join(descriptions) + '.'

    def str2incident(self):
        """
        Convert an undirected graph (given string representation) to incident.
        Input:
            input_str: A string representing the edges of an undirected graph, 
            e.g., "(1,2) (2,3)" indicates there is an edge between node 1 and 2, and between node 2 and 3.
        Output:
            A string describing each node and its connected neighbors in ascending order,
            e.g., 'Node 1 is connected to 2.\nNode 2 is connected to 1, 3.\nNode 3 is connected to 2.'
        """
        adjacency_list = {}
        for edge in self.edges:
            if edge[0] not in adjacency_list:
                adjacency_list[edge[0]] = []
            adjacency_list[edge[0]].append((edge[1], edge[2]))

        for key in adjacency_list.keys():
            adjacency_list[key].sort()

        sorted_adjacency_list = dict(sorted(adjacency_list.items()))
        def func(pair):
            return f'node {pair[0]} with capacity {pair[1]}'

        description = []
        for node, neighbors in sorted_adjacency_list.items():
            description.append(f"Node {node} is connected to {', '.join(map(func, neighbors))}.")
        return "\n".join(description)

    def str2graph_expert(self):
        return super().str2graph()
    
    def str2people(self, relation_str='wrote a paper together'):
        return super().str2people(relation_str)

    def str2nxgraph(self):
        graph = nx.DiGraph()
        for s in self.edges:
            a, b, capacity = s
            capacity = self.number_convert(capacity)
            # print(a, b, capacity)
            graph.add_edge(a, b, capacity=capacity)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # pos=nx.spring_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
        # labels = nx.get_edge_attributes(graph,'capacity')
        # nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels)
        # plt.show()
        return graph

    def get_solution(self):
        _, flow_dict = nx.maximum_flow(self.graph, self.src, self.trg,  capacity='capacity')
        # print(flow_dict)
        from collections import deque
        dq = deque()
        dq.append(self.src)
        solution = ''
        while len(dq):
            head = dq.popleft()
            tmp_str = [f'From node {head}']
            tmp_dict = flow_dict[head]
            for n, v in tmp_dict.items():
                if v != 0:
                    tmp_str.append(f'we can send {round(v, 1)} units of flow to node {n}')
                    # print(n)
                    dq.append(n)
                    flow_dict[head][n] = 0
            if len(tmp_str) == 1:
                continue
            tmp_str = ', '.join(tmp_str) + '. '
            solution += f'\n{tmp_str}'
        return f'{solution}\n'
        
            
    def check_solution(self, response):
        return super().check_solution()