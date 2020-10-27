import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep
import suepsUtilities

plt.style.use(hep.style.ROOT)
boost = False
event = 0

# Get the file and import using uproot
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
#base = '/Users/chrispap/'
# xrootd is not working properly in Python3 :(
base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
datasets = [base +
            'PrivateSamples.SUEP_2018_mMed-%d_mDark-%d_temp-%d_decay-%s'
            '_13TeV-pythia8_n-100_0_RA2AnalysisTree.root'%(mMed, mDark, temp, decayMode),
           ]
rootfile = datasets[0]
fin = uproot.open(rootfile)

# Attach the branches to numpy arrays
tree = fin['TreeMaker2/PreSelection']
def get_branch(branchname):
    return tree[branchname].array()

Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')
HT = get_branch('HT')

jetMultiplicities = [0]
for i in range(Tracks_x.size):
    if HT[i] > 1200:
        event = i

        # Get tracks information
        tracks_x = Tracks_x[event]
        tracks_y = Tracks_y[event]
        tracks_z = Tracks_z[event]
        tracks_E = np.sqrt(tracks_x**2+tracks_y**2+tracks_z**2+0.13957**2)
        tracks = uproot_methods.TLorentzVectorArray.from_cartesian(tracks_x, tracks_y, tracks_z, tracks_E)
        tracks_fromPV0 = Tracks_fromPV0[event]
        tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[event]

        # Select good tracks
        tracks = tracks[(tracks.pt > 1.) &
                        (tracks.eta < 2.5) &
                        (tracks_fromPV0 >= 2) &
                        (tracks_matchedToPFCandidate > 0)]

        jetsAK15 = suepsUtilities.makeJets(tracks, 1.5)
        #print("Event %d:"%event)

        for jet in jetsAK15:
            if jet.pt > 100:
                jetMultiplicities = np.append(jetMultiplicities,[len(jet)])
                #print("  Jet: pt=%f, eta=%f, phi=%f"%(jet.pt, jet.eta, jet.phi))

jetMultiplicities = jetMultiplicities[1:]

fig = plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_xlabel('AK15 jets multiplicity')
ax.set_ylabel('Entries/bin')
plt.hist(jetMultiplicities,bins=50,histtype='step')
plt.show()
