import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm

from mesa import Agent

from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule

from mesa.batchrunner import BatchRunner

from itertools import count
import networkx as nx
from matplotlib import pylab
from matplotlib.pyplot import pause

import time

extremity = 0
quality = 0
confidence = 0
connections = 0

class Opinion():
    def __init__(self, extremity, quality, confidence):
        self.extremity = extremity
        self.quality = quality
        self.confidence = confidence
        
    def low_quality(self):
        return self.quality < -5
    
    def high_quality(self):
        return self.quality > 5
    
    def low_confidence(self):
        return self.confidence < -5
    
    def high_confidence(self):
        return self.confidence > 5

class Citizen(Agent):
    IN_GROUP_RANGE = 2

    def __init__(self, unique_id, model, opinion, connected_citzens=[]):
        super().__init__(unique_id, model)

        self.opinion = opinion
        self.connected_citzens = list(connected_citzens)
        self.connection_strengths = [1 for _ in connected_citzens]
        
        self.planned_extremity = None
        self.planned_quality = None
        self.planned_confidence = None
        self.planned_connections, self.planned_strengths = None, None
    
    def serialize(self):
        return {
            "id": self.unique_id,
            "extremity": self.opinion.extremity,
            "connected_nodes": list(map(lambda x: x.unique_id, self.connected_nodes))
        }
    
    def in_group_citizens(self, pool=None):
        if not pool:
            pool = self.connected_nodes
        
        return list(filter(lambda citizen: abs(citizen.opinion.extremity - self.opinion.extremity) < Citizen.IN_GROUP_RANGE, pool)) 
        # group = []
        # for citizen in pool:
        #     if abs(citizen.opinion.extremity - self.opinion.extremity) < Citizen.IN_GROUP_RANGE:
        #         group.append(citizen)
        # return group
    
    def out_group_citizens(self):
        group = []
        in_group_ids = list(map(lambda x: x.unique_id, self.in_group_citizens()))
        
        for citizen in self.connected_nodes:
            if citizen.unique_id not in in_group_ids:
                group.append(citizen)
        return group
    
    # Plans next state based on perceived behaviour of neighbours
    def update_opinion_extremity(self):
        t0= time.time()
        in_group_opinions = list(map(lambda x: x.opinion.extremity, self.in_group_citizens()))
        out_group_opinions = list(map(lambda x: x.opinion.extremity, self.out_group_citizens()))
           
        all_update_distance = 0
        if len(in_group_opinions) + len(out_group_opinions) > 0:
            all_group_average_extremity = sum(in_group_opinions) + sum(out_group_opinions) / float(len(in_group_opinions) + len(out_group_opinions))                    
            all_update_distance = all_group_average_extremity - self.opinion.extremity / 4.0
        
        out_update_distance = 0
        if len(out_group_opinions) > 0:
            out_group_average_extremity = sum(out_group_opinions) / float(len(out_group_opinions))
            out_update_distance = out_group_average_extremity - self.opinion.extremity / 2.0
        
        new_extremity = 0
        
        if self.opinion.low_quality():
            if self.opinion.low_confidence():
                # Move toward all group average
                new_extremity = self.opinion.extremity + all_update_distance
            else:
                # Move away from out group position
                new_extremity = self.opinion.extremity - out_update_distance
        else:
            if self.opinion.low_confidence():
                # Move toward group average more slowly.
                new_extremity = self.opinion.extremity + all_update_distance / 2.0
            else:
                # No change
                new_extremity = self.opinion.extremity
        
        if new_extremity < -10:
            new_extremity = -10
        elif new_extremity > 10:
            new_extremity = 10
        
        global extremity
        extremity = time.time() - t0
        return new_extremity
    
    def update_opinion_quality(self):
        t0= time.time()
        in_group_opinion_quality = list(map(lambda x: x.opinion.quality, self.in_group_citizens()))
        out_group_opinion_quality = list(map(lambda x: x.opinion.quality, self.out_group_citizens()))
            
#         all_group_average_quality = sum(in_group_opinion_quality) + sum(out_group_opinion_quality) / float(len(in_group_opinion_quality) + len(out_group_opinion_quality))                    
#         all_update_distance = all_group_average_quality - self.opinion.quality / 4.0
        
        in_update_distance = 0
        if len(in_group_opinion_quality) > 0:
            in_group_average_quality = sum(in_group_opinion_quality)/ float(len(in_group_opinion_quality))                    
            in_update_distance = in_group_average_quality - self.opinion.quality / 4.0
        
        global quality
        quality = time.time() - t0
        return self.opinion.quality + in_update_distance
        
    def update_opinion_confidence(self):
        t0= time.time()
        in_group_opinions = len(self.in_group_citizens())
        out_group_opinions = len(self.out_group_citizens())
        
        modifier = 1
        if in_group_opinions + out_group_opinions > 0:
            percent_in_group = in_group_opinions/float(in_group_opinions + out_group_opinions)
            modifier = 1 + (percent_in_group - 0.5)
        
        global confidence
        confidence = time.time() - t0
        return self.opinion.confidence * modifier
        
    def update_connections(self):
        t0= time.time()
        connected_citizens = list(self.connected_nodes)
        connection_strengths = list(self.connection_strengths)
        
        if len(self.connected_nodes) == 0:
            # Should this be random not in-group?
            connection_source = np.random.choice(self.in_group_citizens(pool=self.model.schedule.agents))

        else:
            connection_source = np.random.choice(self.connected_nodes)
            count = 0
            while len(connection_source.connected_nodes) == 0:
                connection_source = np.random.choice(self.connected_nodes)
                count += 1
                if count == len(self.connected_nodes):
                    connection_source = np.random.choice(self.in_group_citizens(pool=self.model.schedule.agents))
                    break

        if len(connection_source.connected_nodes) > 0:
            new_connection = np.random.choice(connection_source.connected_nodes)
        
            if new_connection not in self.connected_nodes:
                connected_citizens.append(new_connection)
                connection_strengths.append(1)

        # for _ in range(1):
        #     second_degree_connections = list(map(lambda x: x.connected_nodes, connected_citizens))
        #     second_degree_connections = [item for sublist in second_degree_connections for item in sublist]
        #     pool = second_degree_connections
            
        #     if len(pool) == 0:
        #         pool = self.in_group_citizens(pool=self.model.schedule.agents)
            
        #     new_node = np.random.choice(pool)
        #     if new_node not in self.connected_nodes:
        #         connected_citizens.append(new_node)
        #         connection_strengths.append(1)
        
        to_remove = []
        
        for other_node in connected_citizens:
            citizen_index = connected_citizens.index(other_node)
            distance = abs(other_node.opinion.extremity - self.opinion.extremity)
            percent_distance = distance/20.0
            connection_strengths[citizen_index] -= percent_distance/2.0
            if connection_strengths[citizen_index] < 0.3:
                to_remove.append(other_node)
            
        for node in to_remove:
            ids = list(map(lambda x: x.unique_id, connected_citizens))
            citizen_index = ids.index(node.unique_id)
            del connected_citizens[citizen_index]
            del connection_strengths[citizen_index]
        
        global connections
        connections = time.time() - t0
        return (connected_citizens, connection_strengths)
                
    # MESA supports simultaneous execution through step and advance. Step is called for all agents before advance.
    # Agents plan in the step phase and then all agents enact their plans in the advance phase.
    
    # REQUIRED METHOD: step is the name used by MESA for the plan stage.
    def step(self):
        self.planned_extremity = self.update_opinion_extremity()
        self.planned_quality = self.update_opinion_quality()
        self.planned_confidence = self.update_opinion_confidence()
        self.planned_connections, self.planned_strengths = self.update_connections()
        
    # REQUIRED METHOD: advance refers to implements planned changes. 
    def advance(self):
        self.opinion.extremity = self.planned_extremity
        self.opinion.quality = self.planned_quality
        self.opinion.confidence = self.planned_confidence
        self.connected_nodes = self.planned_connections
        self.connection_strengths =  self.planned_strengths

class SocietyModel(Model):
    
    # Threshold distribution generators
    def scaled_normal_distribution(n, minimum, maximum):
        mean = (maximum + minimum)/2.0
        sigma = (maximum - mean)/3.0
        return np.array(list(np.random.normal(mean, sigma, (n,))))
    
    # Model intialisation
    def __init__(self,
        num_citizens, connections_per_citizen,
        max_iterations,
        opinion_distribs):

        self.max_iters = max_iterations
        self.iterations = 0
        self.running = True
        self.schedule = SimultaneousActivation(self)
        self.opinion_distributions = {}
        self.num_citizens = num_citizens
        
        self.history = []
        
        for opinion_metric in opinion_distribs:
            ditrib_dict = opinion_distribs[opinion_metric]
            try:
                distrib = ditrib_dict["distrib"]
                vals = distrib.split(",")
                vals = [float(val) for val in vals]
                self.opinion_distributions[opinion_metric] = vals
                self.num_citizens = len(vals)
            except:
                vals = SocietyModel.scaled_normal_distribution(num_citizens, ditrib_dict["minimum"], ditrib_dict["maximum"])
                self.opinion_distributions[opinion_metric] = vals
        
        for citizen_id in range(1, self.num_citizens+1):
            extremity = self.opinion_distributions["extremity"][citizen_id-1]
            quality = self.opinion_distributions["quality"][citizen_id-1]
            confidence = self.opinion_distributions["confidence"][citizen_id-1]

            opinion = Opinion(extremity, quality, confidence)
            self.create_citizen(citizen_id, opinion)
        
        self.connect_citizens(connections_per_citizen)

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={}
        )

    # Creates a store instance and places it on the grid and into the schedule.
    # Location is a random empty cell.
    def create_citizen(self, unique_id, opinion):
        citizen = Citizen(unique_id, self, opinion)
        self.schedule.add(citizen)
        
    def connect_citizens(self, connections_per_citizen):
        population = self.schedule.agents
        for citizen in population:
            citizen.connected_nodes = np.random.choice(population, connections_per_citizen, replace=False)
            citizen.connection_strengths = [1 for _ in citizen.connected_nodes]
    
    # Advance the model one step.
    def step(self):
        if self.running:
            self.datacollector.collect(self)
            self.schedule.step()
            self.iterations += 1
            
            self.history.append(list(map(lambda x: x.serialize(), self.schedule.agents)))
            
            if self.iterations > self.max_iters:
                self.running = False


# In[4]:

def run_model(config, plot=False):
    print("\n========\nRUNNING\n========")
    model = SocietyModel(**config)

    while model.running:   
        model.step()

    print("\n========\nRESULTS\n========")
    if plot:
        plt.subplot(321)
        plt.title("Extremity - Pre")
        plt.hist(model.opinion_distributions["extremity"])
        
        extremity = list(map(lambda x: x.opinion.extremity, model.schedule.agents))
        plt.subplot(322)
        plt.title("Extremity - Post")
        plt.hist(extremity)
        
        plt.subplot(323)
        plt.title("Quality - Pre")
        plt.hist(model.opinion_distributions["quality"])
        
        extremity = list(map(lambda x: x.opinion.quality, model.schedule.agents))
        plt.subplot(324)
        plt.title("Quality - Post")
        plt.hist(extremity)
        
        plt.subplot(325)
        plt.title("Confidence - Pre")
        plt.hist(model.opinion_distributions["confidence"])
        
        extremity = list(map(lambda x: x.opinion.confidence, model.schedule.agents))
        plt.subplot(326)
        plt.title("Confidence - Post")
        plt.hist(extremity)
        
        plt.tight_layout()
        plt.show()
        
    return model
        
        


# In[5]:

def get_config():        
    return {
        "num_citizens": 500,
        "connections_per_citizen": 10,
        "opinion_distribs": {
            "extremity": {"minimum": -3, "maximum": 3},
            "quality": {"minimum": -7, "maximum": 3},
            "confidence": {"minimum": 0, "maximum": 4},
        },
        "max_iterations": 100,
    }

history = None
def main():
    global history
    config = get_config()
    model = run_model(config, plot=False)
    history = model.history
    print("extremity " + str(extremity))
    print("quality " + str(quality))
    print("confidence " + str(confidence))
    print("connections " + str(connections))
main()


# In[6]:

def network_plot(data, img_name):
    g = nx.Graph()
    color_map = []
    i = data
    for nodes in range(len(i)):
        g.add_node(i[nodes]["id"], extremity=float(i[nodes]["extremity"]))
    groups = set(nx.get_node_attributes(g,'extremity').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = g.nodes()
    colors = [mapping[g.node[n]['extremity']] for n in nodes]
    for a in range(len(i)):
        for con in range(len(i[a]["connected_nodes"])):
            g.add_edge(i[a]["id"],i[a]["connected_nodes"][con])
    plot = nx.draw(g, pos=nx.spring_layout(g), node_color=colors, cmap=plt.cm.seismic)
    plt.show()
    fig = plt.figure()


def network_plot(history):
    pylab.ion()

    def get_fig(data):
        g = nx.Graph()
        color_map = []
        i = data
        for nodes in range(len(i)):
            g.add_node(i[nodes]["id"], extremity=float(i[nodes]["extremity"]))
        groups = set(nx.get_node_attributes(g,'extremity').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = g.nodes()
        colors = [mapping[g.node[n]['extremity']] for n in nodes]
        for a in range(len(i)):
            for con in range(len(i[a]["connected_nodes"])):
                g.add_edge(i[a]["id"],i[a]["connected_nodes"][con])
        
        fig = pylab.figure()
        nx.draw(g, pos=nx.spring_layout(g), node_color=colors, cmap=plt.cm.seismic)
        return fig

    pylab.show()

    for data in history:
        fig = get_fig(data)
        fig.canvas.draw()
        pylab.draw()
        pause(0.2)
        pylab.close(fig)



