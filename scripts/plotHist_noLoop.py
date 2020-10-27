import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

# Input section  -  You may want to edit these
# File selection
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
base = '/Users/chrispap/'
# xrootd is not working properly in Python3 :(
#base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
datasets = [base +
            'PrivateSamples.SUEP_2018_mMed-%d_mDark-%d_temp-%d_decay-%s'
            '_13TeV-pythia8_n-100_0_RA2AnalysisTree.root'%(mMed, mDark, temp, decayMode),
           ]

# Switch to true if you want to boost along the scalar 4-momentum
boost = True
# Switch to true to save the figure as a PDF
save = True

# Get the file and import using uproot
rootfile = datasets[0]
fin = uproot.open(rootfile)
# Attach the branches to numpy arrays
tree = fin['TreeMaker2/PreSelection']
def get_branch(branchname):
    return tree[branchname].array()

HT = get_branch('HT')

GenParticles_pt = get_branch('GenParticles.fCoordinates.fPt')
GenParticles_eta = get_branch('GenParticles.fCoordinates.fEta')
GenParticles_phi = get_branch('GenParticles.fCoordinates.fPhi')
GenParticles_E = get_branch('GenParticles.fCoordinates.fE')
GenParticles_ParentId = get_branch('GenParticles_ParentId')
GenParticles_PdgId = get_branch('GenParticles_PdgId')
GenParticles_Status = get_branch('GenParticles_Status')
GenParticles = uproot_methods.TLorentzVectorArray.from_ptetaphie(GenParticles_pt,
                                                                 GenParticles_eta,
                                                                 GenParticles_phi,
                                                                 GenParticles_E)
GenParticles = GenParticles[HT > 1200]
GenParticles_PdgId = GenParticles_PdgId[HT > 1200]
GenParticles_ParentId = GenParticles_ParentId[HT > 1200]
GenParticles_Status = GenParticles_Status[HT > 1200]
# Define mask arrays to select the desired particles
FinalParticles = GenParticles[(GenParticles_Status == 1) & (GenParticles.pt > 1) & (np.abs(GenParticles.eta) < 2.5)]
FinalParticles_ParentId = GenParticles_ParentId[(GenParticles_Status == 1) & (GenParticles.pt > 1) & (np.abs(GenParticles.eta) < 2.5)]
FromScalarParticles = FinalParticles[FinalParticles_ParentId == 999998]
IsrParticles = FinalParticles[FinalParticles_ParentId != 999998]

Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')
Tracks_E = np.sqrt(Tracks_x**2+Tracks_y**2+Tracks_z**2+0.13957**2)
Tracks = uproot_methods.TLorentzVectorArray.from_cartesian(Tracks_x, Tracks_y, Tracks_z, Tracks_E)
# Select good tracks
Tracks = Tracks[(Tracks.pt > 1.) & (np.abs(Tracks.eta) < 2.5) & (Tracks_fromPV0 >= 2) &
                (Tracks_matchedToPFCandidate > 0)]

# Plot results
fig = plt.figure(figsize=(8,8))
ax = plt.gca()
#ax.set_yscale('log')
#ax.set_xscale('log')

#ax.hist(FromScalarParticles.energy.mean(),bins=50, histtype='step', label='from scalar', color='r')
#ax.hist(IsrParticles.energy.mean(),bins=50, histtype='step', label='from ISR', color='b')
ax.hist(FromScalarParticles.energy.max()/IsrParticles.energy.mean(),bins=100, histtype='step', color='b')
ax.set_xlabel('Event average energy ratio')

plt.legend()
plt.show()
