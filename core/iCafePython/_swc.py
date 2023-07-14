#submodule of iCafe for traces load write
import os
from .swcnode import SWCNode
from .snake import Snake
from .snakelist import SnakeList
from .definition import BOITYPENUM, VESTYPENUM, matchvestype
import copy
import numpy as np
from .utils.swc_utils import loadSWCFile, writeSWCFile, getUniqueSWCFromPtlist
from rich import print


def loadSWC(self, swctype='raw_ves'):
    swcfilename = os.path.join(self.path, 'tracing_' + swctype + '_TH_' + self.filename_solo + '.swc')
    self._snakelist, self._swclist = loadSWCFile(swcfilename)
    return self._snakelist, self._swclist


def readSnake(self, swctype='raw_ves', root_results=None):
    swcfilename = os.path.join(self.path if root_results is None else root_results,
                               'tracing_' + swctype + '_TH_' + self.filename_solo + '.swc')
    snakelist, _ = loadSWCFile(swcfilename)
    return snakelist


def writeSWC(self, swc='ai_ves', snakelist=None, path=None):
    if path is None:
        path = os.path.join(self.path, 'tracing_' + swc + '_TH_' + self.filename_solo + '.swc')
    if snakelist is None:
        snakelist = self.snakelist
    writeSWCFile(path, snakelist)


def loadVes(self, src='ves', mode='r', vesfilename=None):
    if vesfilename is None:
        vesfilename = os.path.join(self.path, 'tracing_' + src + '_TH_' + self.filename_solo + '.swc')
    if mode == 'u':
        #no duplicate ves point
        return self._loadVesNoDuplicate(vesfilename)
    elif mode == 'r':
        #raw ves file
        return self._loadVesNoChange(vesfilename)


def _loadVesNoDuplicate(self, path):
    if not os.path.exists(path):
        raise FileNotFoundError('No vessel file available', path)
    self._veslist = [[] for i in range(VESTYPENUM)] #list, first of vessel type, then each snake in that type
    self._vessnakelist = SnakeList() #everything in one list
    cveslist = []
    starttype = -1
    endtype = -1

    self.ptlist = [] #unique id
    self.ptvesid = [] #from unique id to veslist id
    IDMap = [0] #from ves file id to unique id, starting from 1
    self.deglist = {} #degree of nodes, from unique id to degree val

    with open(path, 'r') as fp:
        for line in fp:
            cswcnode = SWCNode.fromline(line)
            cswcnodeori = copy.deepcopy(cswcnode)
            uniqueswcnode = getUniqueSWCFromPtlist(self.ptlist, cswcnode)
            IDMap.append(uniqueswcnode.id)
            #update pid
            if uniqueswcnode.pid != -1:
                uniqueswcnode.pid = IDMap[uniqueswcnode.pid]
            cveslist.append(uniqueswcnode)
            #if new, not change
            if uniqueswcnode.id == cswcnode.id:
                self.ptlist.append(uniqueswcnode)
                self.ptvesid.append(len(self._vessnakelist))
                if uniqueswcnode.id not in self.deglist:
                    self.deglist[uniqueswcnode.id] = []

            if uniqueswcnode.pid != -1:
                if uniqueswcnode.pid not in self.deglist[uniqueswcnode.id]:
                    self.deglist[uniqueswcnode.id].append(uniqueswcnode.pid)
                if uniqueswcnode.id not in self.deglist[uniqueswcnode.pid]:
                    self.deglist[uniqueswcnode.pid].append(uniqueswcnode.id)
            #if uniqueswcnode.id==15 or uniqueswcnode.pid==15:
            #	print(cswcnode,uniqueswcnode,self.deglist[15])

            if cswcnode.type != 0:
                if starttype == -1:
                    starttype = cswcnode.type
                else:
                    endtype = cswcnode.type
                    cvestype = matchvestype(starttype, endtype)
                    starttype = -1
                    endtype = -1
                    if cvestype != -1:
                        self._veslist[cvestype].append(copy.copy(cveslist))
                        self._vessnakelist.append(Snake(cveslist, cvestype))
                    else:
                        print('Unknown ves match type', starttype, endtype)
                    cveslist.clear()
    return self._vessnakelist


def _loadVesNoChange(self, path):
    if not os.path.exists(path):
        raise FileNotFoundError('No vessel file available', path)
    self._vessnakelist = SnakeList()
    cveslist = []
    starttype = -1
    endtype = -1

    with open(path, 'r') as fp:
        for line in fp:
            cswcnode = SWCNode.fromline(line)
            cveslist.append(cswcnode)
            if cswcnode.type != 0:
                if starttype == -1:
                    starttype = cswcnode.type
                else:
                    endtype = cswcnode.type
                    cvestype = matchvestype(starttype, endtype)
                    starttype = -1
                    endtype = -1
                    if cvestype != -1:
                        self._vessnakelist.addSnake(Snake(cveslist, cvestype))
                    else:
                        print('Unknown ves match type', starttype, endtype)
                    cveslist.clear()
    self._veslist = self._vessnakelist.toVesList()
    return self._vessnakelist


def matchVesFromSnake(self, include_list=None):
    if include_list is None:
        include_list = np.arange(VESTYPENUM).tolist()
    sel_snake_ids = []
    #gather points on the ves within include list to a set
    ves_pts = set()
    for vesi in range(self.vessnakelist.NSnakes):
        if self.vessnakelist[vesi].type not in include_list:
            continue
        for pti in range(self.vessnakelist[vesi].NP):
            pos = self.vessnakelist[vesi][pti].pos.hashPos()
            ves_pts.add(pos)
    #search each snakelist to find start and end points for selection
    for snakei in range(self.snakelist.NSnakes):
        startid = -1
        endid = self.snakelist[snakei].NP
        for pti in range(self.snakelist[snakei].NP):
            pos = self.snakelist[snakei][pti].pos.hashPos()
            if pos in ves_pts:
                if startid == -1:
                    startid = pti
            else:
                if startid != -1:
                    endid = pti
                    if endid - startid > 2:
                        sel_snake_ids.append((snakei, startid, endid))
                    startid = -1
                    endid = self.snakelist[snakei].NP
        if startid != -1 and endid - startid > 2:
            sel_snake_ids.append((snakei, startid, endid))
    return sel_snake_ids
