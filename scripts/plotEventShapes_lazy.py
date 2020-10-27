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

variable = 'sphericity'
#variable = 'aplanarity'
#variable = 'C'
#variable = 'D'
#variable = 'circularity'
#variable = 'isotropy'

# Get the file and import using uproot
# QCD parameters
htBins = ['1000to1500','1500to2000','2000toInf']
base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
datasets = [
            base + 'Autumn18.QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
           ]

# Attach the branches to numpy arrays
def get_branch(branchname):
    return uproot.lazyarray(datasets, 'TreeMaker2/PreSelection', branchname)

HT = get_branch('HT')
CrossSection = get_branch('CrossSection')
Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')
print("Got all branches!")

evtShape = np.zeros(Tracks_x.size)
for ievt in range(Tracks_x.size):
    if ievt%1000 == 0:
        print("Processing event %d. Progress: %.2f%%"%(ievt,100*ievt/Tracks_x.size))
    if HT[ievt] < 1200:
        evtShape[ievt] = -1
        continue
    tracks_x = Tracks_x[ievt]
    tracks_y = Tracks_y[ievt]
    tracks_z = Tracks_z[ievt]
    tracks_E = np.sqrt(tracks_x**2+tracks_y**2+tracks_z**2+0.13957**2)
    tracks = uproot_methods.TLorentzVectorArray.from_cartesian(tracks_x, tracks_y, tracks_z, tracks_E)
    tracks_fromPV0 = Tracks_fromPV0[ievt]
    tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[ievt]
    tracks = tracks[(tracks.pt > 1.) & (tracks.eta < 2.5) & (tracks_fromPV0 >= 2) & (tracks_matchedToPFCandidate > 0)]
    jetsAK15 = suepsUtilities.makeJets(tracks, 1.5)
    isrJet = suepsUtilities.isrTagger(jetsAK15,warn=False)
    tracks_boosted_minusISR = tracks.boost(-isrJet.p3/isrJet.energy)
    s = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR)
    evtShape[ievt] = eventShapesUtilities.sphericity(s)

# Plot results
fig = plt.figure(figsize=(8,8))
ax = plt.gca()

CrossSection = CrossSection[HT > 1200]
evtShape = evtShape[HT > 1200]

ax.hist(evtShape, bins=100, density=True, weights=CrossSection, histtype='step', color='b')

plt.show()
