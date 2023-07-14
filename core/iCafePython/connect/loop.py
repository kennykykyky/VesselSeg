import networkx as nx
import matplotlib.pyplot as plt

def addBranchEdge(snakelist,branch_G,thres_dist=10):
    for snakei in range(snakelist.NSnakes):
        #check head and tails
        for ckpt in [0,-1]:
            cpos = snakelist[snakei][ckpt].pos
            match_snakeid, match_ptid, match_dist, match_rad = snakelist.matchPt(cpos,snakei,thres_dist)
            if match_snakeid!=-1:
                if match_dist<thres_dist:
                    print(snakei,'head' if ckpt==0 else 'tail','match to',match_snakeid,'dist',match_dist)
                    closs = -match_dist
                    branch_G.add_edge(snakei,match_snakeid,dist=match_dist,loss=closs,snakei=snakei,pti=ckpt,snakej=match_snakeid,ptj=match_ptid)
    return branch_G

def constructSnakeFromTree(snakelist,tree_G):
    snakelist = snakelist.copy()
    for edgei in tree_G.edges():
        edge_item = tree_G.edges[edgei]
        snakei, pti, snakej, ptj = edge_item['snakei'],edge_item['pti'],edge_item['snakej'],edge_item['ptj']
        if edge_item['dist']!=0:
            print('branch',snakei, pti, snakej, ptj,edge_item['dist'])
            snakelist[snakei].branchSnake(snakelist[snakej][ptj], pti)
    #snakelist.autoMerge()
    #snakelist = snakelist.mainArtTree(10,5)
    return snakelist

def cutLoop(seg_ves_snakelist):
    DEBUG = 1
    seg_ves_snakelist.autoBranch()
    seg_ves_snakelist.removeDuplicatePts()
    seg_ves_snakelist.assignDeg()
    seg_ves_snakelist.autoTransform()
    seg_ves_snakelist.assignDeg()

    branch_G = seg_ves_snakelist.branchGraph()
    branch_G = addBranchEdge(seg_ves_snakelist, branch_G, 17)
    if DEBUG:
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(branch_G, font_size=13, node_size=[20 + branch_G.nodes[i]['NP'] for i in branch_G.nodes()],
                         pos={i: branch_G.nodes[i]['pos'][:2] for i in branch_G.nodes()}, node_color='r')
        plt.show()
    tree_G = nx.minimum_spanning_tree(branch_G, 'loss')
    if DEBUG:
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(tree_G, font_size=13, node_size=[20 + branch_G.nodes[i]['NP'] for i in branch_G.nodes()],
                         pos={i: tree_G.nodes[i]['pos'][:2] for i in tree_G.nodes()}, node_color='r')
        plt.show()
    print('prev edges',len(tree_G.edges()),'new edges',len(branch_G.edges()))
    seg_ves_snakelist = constructSnakeFromTree(seg_ves_snakelist, tree_G)
    main_tree_snakelist = seg_ves_snakelist.mainArtTree(10, 15)
    main_tree_snakelist.autoMerge()
    return main_tree_snakelist
