import argparse
import csv
import os
import math
import numpy as np

from plot import PlotLinesHandler

class ArgsModel(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--N", type=int, default=20,
            help="group size")
        self.parser.add_argument("--n_repl", type=int, default=50,
            help="the # of replications")
        self.parser.add_argument("--n_gen", type=int, default=1000,
            help="the # of generations")
        self.parser.add_argument("--n_trial", type=int, default=10,
            help="the # of trails per generation")
        self.parser.add_argument("--res", type=int, default=10,
            help="the # of resources that each actor is given in each trial")
        self.parser.add_argument("--val_ratio", type=float, default=2.0,
            help="value of resources that each actor receives from others")
        self.parser.add_argument("--mut_ratio", type=float, default=0.05,
            help="mutation rate")
        self.parser.add_argument("--giving_init", type=str, default="random",
            help="\"all_0\" for all giving genes are set 0; \"random\" for all are set randomly from [0, 10].")
        
        self.parser.add_argument("--simuNo", type=float, default=2.0,
            help="1.1 for simulation I-1; 1.2 for simulation I-2; 2 for simulation II.")
        self.parser.add_argument("--rnd_seed", type=int, default=202201,
            help="random seed.")
    

    @staticmethod
    def set_simu_param(args:argparse.ArgumentParser, simuNo:float) -> dict:
        """ Set parameters for the simulation. """
        args.simuNo = simuNo
        if int(simuNo) == 1:
            args.N = 20
            args.n_repl = 50
            args.n_gen = 1000
            args.n_trial = 10
            args.res = 10
            args.val_ratio = 2.0
            args.mut_ratio = 0.05
            if simuNo == 1.1:
                args.giving_init = "random"
            elif simuNo == 1.2:
                args.giving_init = "all_0"
            else:
                ValueError("simuNo unknowned.")
            
            return args
        
        elif simuNo == 2.0:
            args.N = 100
            args.n_repl = 50
            args.n_gen = 200
            args.n_trial = 10
            args.res = 10
            args.val_ratio = 2.0
            args.mut_ratio = 0.01
            args.giving_init = "all_0"
            return args

        return args
    

    def get_args(self) -> argparse.ArgumentParser:
        args = self.parser.parse_args()
        return self.set_simu_param(args, args.simuNo)    
    

    def get_simu_args(self, simuNo:float) -> argparse.ArgumentParser:
        args = self.parser.parse_args()
        return self.set_simu_param(args, simuNo)


class Agent:

    def __init__(self, giving_init, id) -> None:
        self.id = id
        self.giving = None
        self.tolerance = None
        self.profit = 0

        if giving_init == "random":
            self.set_giving()
        elif giving_init == "all_0":
            self.set_giving(0)
        else:
            raise ValueError("Unknown giving_init when initializing agents' giving genes.")
        self.set_tolerance()
    
    def get_M(self):
        return self.giving * self.tolerance

    def set_giving(self, val=None):
        """ randomly select from [0, 10] if val is not given. """
        if val == None:
            self.giving = np.random.randint(0, 11)
        else:
            self.giving = val
    
    def set_tolerance(self):
        """ randomly (uniformly) select from [0.1, 2.0]. """
        self.tolerance = np.random.uniform()*1.9 + 0.1
    
    def copy_genes(self, ag):
        self.giving = ag.giving
        self.tolerance = ag.tolerance
    
    def reset_profit(self):
        self.profit = 0
    

class Game:
    DIRECTION = ((-1, -1),
                 (-1, 0),
                 (-1, 1),
                 (0, -1),
                 (0, 1),
                 (1, -1),
                 (1, 0),
                 (1, 1))
    
    def __init__(self, args, random_seed, verbose=True) -> None:
        np.random.seed(random_seed)
        self.verbose = verbose
        self.args = args
        if self.verbose:
            print(args)

        self.N = args.N
        self.ags = [Agent(args.giving_init, id) for id in range(args.N)]

        if int(self.args.simuNo) == 1:
            # making a global recipient list
            self.sorted_ags = sorted(self.ags, key=lambda ag: ag.giving)
            self.giving_ind_list, self.max_ind = self._set_index_list(self.sorted_ags, self.N)
        elif int(self.args.simuNo) == 2:
            # building a net for each agent
            assert self.N == 100
            self.ags_net = list()
            for i in range(10):
                for j in range(10):
                    nei_list = list()
                    for di, dj in Game.DIRECTION:
                        if 0 <= i + di < 10 and 0 <= j + dj < 10:
                            nei_list.append(self.ags[(i+di)*10+(j+dj)])
                    self.ags_net.append(nei_list)

        self.giv_list = list()
        self.tol_list = list()
    

    @staticmethod
    def _set_index_list(sorted_ags, N):
        """ 
        Return an index list "ind", and the index where the second best giving_genes value first appeared.
        ind: giving_genes with value i start from index ind[i] in sorted_ags, i \in [0, 10]; len(ind)=11.
        """
        ctr = sorted_ags[0].giving
        ind = [0] * sorted_ags[0].giving
        ind += [N] * (11 - sorted_ags[0].giving)
        for ag_idx in range(N):
            while sorted_ags[ag_idx].giving >= ctr:
                ind[ctr] = ag_idx
                ctr += 1
        return ind, ind[ctr-1]
    

    def choose_recipient(self, ag_idx: int) -> list:
        """ Return a list of agents. """

        ag_M = math.ceil(self.ags[ag_idx].get_M())
        chosen_ag_idx = None

        if int(self.args.simuNo) == 1:
            # no agents fit criterion, choose second best.
            if ag_M > 10 or self.giving_ind_list[ag_M] == self.N:
                chosen_ag_idx = np.random.randint(self.max_ind, self.N, size=self.args.n_trial)
            
            # a list of agents that fit criterion
            else:
                chosen_ag_idx = np.random.randint(self.giving_ind_list[ag_M], self.N, size=self.args.n_trial)
            
            return [self.sorted_ags[i] for i in chosen_ag_idx]
        
        elif int(self.args.simuNo) == 2:
            sorted_net = sorted(self.ags_net[ag_idx], key=lambda ag: ag.giving)
            giving_ind_list, max_ind = self._set_index_list(sorted_net, len(sorted_net))

            # no agents fit criterion, choose second best.
            if ag_M > 10 or giving_ind_list[ag_M] == len(sorted_net):
                chosen_ag_idx = np.random.randint(max_ind, len(sorted_net), size=self.args.n_trial)
            
            # a list of agents that fit criterion
            else:
                chosen_ag_idx = np.random.randint(giving_ind_list[ag_M], len(sorted_net), size=self.args.n_trial)
            
            return [sorted_net[i] for i in chosen_ag_idx]
    

    @staticmethod
    def _draw(p):
        return 1 if np.random.uniform() < p else 0
    

    def simulate(self, log_v=50):
        for gen_idx in range(self.args.n_gen):
            if gen_idx % log_v == 0 and self.verbose:
                print("Generation {}/{}".format(gen_idx+1, self.args.n_gen))
            
            for ag in self.ags:
                ag.reset_profit()
            for ag_idx in range(self.N):
                for recei_ag in self.choose_recipient(ag_idx):
                    recei_ag.profit += ag.giving * self.args.val_ratio
            
            # natural selection
            if int(self.args.simuNo) == 1:
                profit_arr = np.array([ag.profit for ag in self.ags])
                profit_mean, profit_sd = np.mean(profit_arr), np.std(profit_arr)
                popped_idx = [ag.id for ag in self.ags if ag.profit < profit_mean - profit_sd]
                insert_idx = [ag.id for ag in self.ags if ag.profit > profit_mean + profit_sd]
                
                # make sure the # of the popped equals to the # of the inserted
                ## not sure how
                if len(popped_idx) != len(insert_idx):
                    profit_sorted = sorted(self.ags, key=lambda ag: ag.profit)
                    if len(popped_idx) > len(insert_idx):
                        insert_idx = [profit_sorted[idx].id for idx in range(self.N-len(popped_idx), self.N)]
                    elif len(popped_idx) < len(insert_idx):
                        popped_idx = [profit_sorted[idx].id for idx in range(0, len(insert_idx))]
                
                for i in range(len(insert_idx)):
                    self.ags[popped_idx[i]].copy_genes(self.ags[insert_idx[i]])
            
            elif int(self.args.simuNo) == 2:
                for ag_idx in range(self.N):
                    profit_sorted = sorted(self.ags_net[ag_idx], key=lambda ag: ag.profit)
                    if profit_sorted[-1].profit > self.ags[ag_idx].profit:
                        self.ags[ag_idx].copy_genes(profit_sorted[-1])
            
            # mutation
            for ag in self.ags:
                if self._draw(self.args.mut_ratio):
                    ag.set_giving()
                if self._draw(self.args.mut_ratio):
                    ag.set_tolerance()
            
            if int(self.args.simuNo) == 1:
                self.sorted_ags = sorted(self.ags, key=lambda ag: ag.giving)
                self.giving_ind_list, self.max_ind = self._set_index_list(self.sorted_ags, self.N)

            # record results
            self.push_means_intoList()
            

    def push_means_intoList(self):
        mean_giving = np.mean(np.array([ag.giving for ag in self.ags]))
        mean_tolerance = np.mean(np.array([ag.tolerance for ag in self.ags]))
        self.giv_list.append(mean_giving)
        self.tol_list.append(mean_tolerance)
    

    def get_result_list(self):
        return self.giv_list, self.tol_list
    

    def print_ags(self):
        print("====================")
        print("ag_idx\t profit\t giving")
        for ag in self.ags:
            print("{}\t {}\t {}".format(ag.id, ag.profit, ag.giving))
        



if __name__ == "__main__":
    args_hdl = ArgsModel()
    args = args_hdl.get_args()
    print(args)

    giv_arr = list()
    tol_arr = list()
    for repli_idx in range(args.n_repl):
        print("replication {}/{} | rnd_seed {}".format(repli_idx+1, args.n_repl, args.rnd_seed+repli_idx))
        game = Game(args, args.rnd_seed+repli_idx, verbose=False)
        game.simulate()
        giv_list, tol_list = game.get_result_list()
        giv_arr.append(giv_list)
        tol_arr.append(tol_list)
    
    giv_arr = np.array(giv_arr)
    tol_arr = np.array(tol_arr)
    
    print("shape: {}".format(giv_arr.shape))
    print("giving   : ({}, {})".format(np.mean(np.array(giv_arr[:, -1])), np.std(np.array(giv_arr[:, -1]))))
    print("tolerance: ({}, {})".format(np.mean(np.array(tol_arr[:, -1])), np.std(np.array(tol_arr[:, -1]))))

    # plot
    plot_hdl = PlotLinesHandler(xlabel="Generation", ylabel="Giving", x_lim=args.n_gen,
                                ylabel_show="Giving")
    plot_hdl.plot_line(np.mean(giv_arr, axis=0), legend="Giving")
    plot_hdl.plot_line(np.mean(tol_arr, axis=0), legend="Tolerance")
    plot_hdl.save_fig(title_param="simu_{}_rndSeed_{}_Nrepl_{}_Giv_{}_{}_Tol_{}_{}".format(args.simuNo, args.rnd_seed, args.n_repl,
        np.mean(np.array(giv_arr[:, -1])), np.std(np.array(giv_arr[:, -1])),
        np.mean(np.array(tol_arr[:, -1])), np.std(np.array(tol_arr[:, -1]))))

