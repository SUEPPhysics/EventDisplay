import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep
import pyjet
import eventShapesUtilities
import suepsUtilities
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.style.use(hep.style.ROOT)

htBins = ['1000to1500','1500to2000','2000toInf']
base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
datasets = [
            base + 'Autumn18.QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
           ]

def get_branch(branchname):
    return uproot.lazyarray(datasets, 'TreeMaker2/PreSelection', branchname)

HT = get_branch('HT')
Weight = get_branch('Weight')
CrossSection = get_branch('CrossSection')

fig = plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_yscale('log')

ax.hist(np.array(HT), bins=100, density=True, weights=Weight, histtype='step', color='r')
ax.hist(np.array(HT), bins=100, density=True, weights=CrossSection, histtype='step', color='b')

plt.show()
