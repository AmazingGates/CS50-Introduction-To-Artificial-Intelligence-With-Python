# Search - This where we will start to apply code to show examples of Search in action.

import sys

class Node(): # This is where we create and define our Node class
    def __init__(self, state, parent, action): # This is our constructor method with parameters
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier(): # This is where we create and define our StackFrontier class
    def __init__(self): # This is our constructor method with parameter
        self.frontier = [] # This is our frontier represented by an empty list / []

    def add(self, node): # This is our add function, that will add something to the 
        #frontier by appending it to the end of the list
        self.frontier.append(node) # This is how we append to our list / []

    def contains_state(self, state): # This function will check if the frontier contains a particular state
        return any(node.state == state for node in self.frontier)
    
    def empty(self): # This function will check if the frontier is empty. That just means that the length of
        #the frontier is 0.
        return len(self.frontier) == 0
    
    def remove(self): # This function will remove something from our frontier. We CAN'T remove something 
        #if the frontier is empty.
        if self.empty():
            raise Exception("Empty Frontier")
        else: # Otherwise, since we are using a Stack (last-in first-out data structure), which means the last
            #thing we added to the frontier (the last thing in our list / []) is the item that we should remove
            #from the frontier
            node = self.frontier[-1] # This is how we remove the last item from our list / []. [-1] represents
            #the last item in our list / []
            self.frontier = self.frontier[:-1] # This is where we update the node to say go head and remove that 
            #node that we just removed from the frontier
            return node # This is where we return the updated node
        

# This is where we will code an alternative version of the code above, in the form of a QueueFrontier, which
#will inherit properties from a StacKFrontier parent class. This means that our QueueFrontier will do 
#everything that our StackFrontier did. Except, the way we remove something is going to be slightly different.
class QueueFrontier(StackFrontier):
    def remove(self): # In the QueueFrontier, instead of removing from the end of the list / [], we are going
        #to start at the beginning of the list / []
        if self.empty():
            raise Exception("Empty Frontier")
        else: # Otherwise, since we are using a Stack (last-in first-out data structure), which means the last
            #thing we added to the frontier (the last thing in our list / []) is the item that we should remove
            #from the frontier
            node = self.frontier[0] # This is how we remove the first item from our list / []. [0] represents
            #the first item in our list / []
            self.frontier = self.frontier[1:] # This is where we update the node to say go head and remove that 
            #node that we just removed from the frontier
            return node # This is where we return the updated node
        
class Maze(): # This is where we create and define a Maze class. This is going to handle the process of taking
    #a sequence, a maze-like text file, and figuring out how to solve it.
    def __init__(self, filename): # This is our constructor method. It will take a filename as an input

        # This is where we read file and set height and width of maze
        with open(filename) as f: 
            contents = f.read()

        # This where we Validate start and goal
        if contents.count("A") != 1:
            raise Exception("Maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("Maze must have exactly one goal")
        
        # This is where we Deteremine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # This is where Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i,j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None


    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("#", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()


    def neighbors(self, state):
        row, col = state

    # This is where we will have all possible Actions
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        # This is where we ensure actions are valid
        result = []
        for action, (r, c) in candidates:
            try:
                if not self.walls[r][c]:
                    result.append((action, (r, c)))
            except IndexError:
                continue
        return result
    

    def solve(self):
        """Finds a solution to maze, if one exists."""

        # This is where we keep track of number of states explored
        self.num_explored = 0

        # This is where we initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None)
        frontier = StackFrontier()
        frontier.add(start)

        # This is where we initialize an empty explored set
        self.explored = set()

        # This is where we keep looping until a solution is found
        while True:

            # If nothing is left in frontier, there is no path
            if frontier.empty():
                raise Exception("No Solution")
            
            # This is where we choose a node from the frontier
            node = frontier.remove()
            self.num_explored += 1

            # This is where we determine if node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []

                # This is where we follow parent nodes to find solution
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return
           
            # This is where we mark node as explored
            self.explored.add(node.state)

            # This is where we add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)


#output_image(self, filename, show_solution, show_explored=False):
from PIL import Image, ImageDraw


