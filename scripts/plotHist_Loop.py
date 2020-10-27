import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep
import suepsUtilities

plt.style.use(hep.style.ROOT)

# Input section  -  You may want to edit these
# File selection
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
base = '/Users/chrispap/'
datasets = [base +
            'PrivateSamples.SUEP_2018_mMed-%d_mDark-%d_temp-%d_decay-%s'
            '_13TeV-pythia8_n-100_0_RA2AnalysisTree.root'%(mMed, mDark, temp, decayMode),
           ]

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
# The last copy of the scalar mediator
ScalarParticle = GenParticles[(GenParticles_PdgId == 25) & (GenParticles_Status == 62)]

Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')
Tracks_E = np.sqrt(Tracks_x**2+Tracks_y**2+Tracks_z**2+0.13957**2)
Tracks = uproot_methods.TLorentzVectorArray.from_cartesian(Tracks_x, Tracks_y, Tracks_z, Tracks_E)
# Select good tracks
Tracks = Tracks[HT > 1200]
Tracks_fromPV0 = Tracks_fromPV0[HT > 1200]
Tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[HT > 1200]
Tracks = Tracks[(Tracks.pt > 1.) & (np.abs(Tracks.eta) < 2.5) & (Tracks_fromPV0 >= 2) &
                (Tracks_matchedToPFCandidate > 0)]

for ievt in range(Tracks.size):
    scalarParticle = ScalarParticle[ievt]

    # Tracks in the event
    tracks = Tracks[ievt]

    # Cluster AK15 jets
    jetsAK15 = suepsUtilities.makeJets(tracks, 1.5)
    isrJet = suepsUtilities.isrTagger(jetsAK15)

    # Boost everything to scalar's rest frame
    print('Event %d'%ievt)
    print(isrJet.p3/isrJet.energy-scalarParticle.p3/scalarParticle.energy)
