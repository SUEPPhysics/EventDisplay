import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep
import pyjet
import eventShapesUtilities

plt.style.use(hep.style.ROOT)
boost = False

# Get the file and import using uproot
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
base = '/Users/chrispap/'
#base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
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

GenParticles_pt = get_branch('GenParticles.fCoordinates.fPt')
GenParticles_eta = get_branch('GenParticles.fCoordinates.fEta')
GenParticles_phi = get_branch('GenParticles.fCoordinates.fPhi')
GenParticles_E = get_branch('GenParticles.fCoordinates.fE')
GenParticles_ParentId = get_branch(b'GenParticles_ParentId')
GenParticles_PdgId = get_branch(b'GenParticles_PdgId')
GenParticles_Status = get_branch(b'GenParticles_Status')
HT = get_branch(b'HT')

Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')

GenParticles_pt = GenParticles_pt[HT > 1200]
GenParticles_eta = GenParticles_eta[HT > 1200]
GenParticles_phi = GenParticles_phi[HT > 1200]
GenParticles_E = GenParticles_E[HT > 1200]
GenParticles_ParentId = GenParticles_ParentId[HT > 1200]
GenParticles_PdgId = GenParticles_PdgId[HT > 1200]
GenParticles_Status = GenParticles_Status[HT > 1200]
Tracks_x = Tracks_x[HT > 1200]
Tracks_y = Tracks_y[HT > 1200]
Tracks_z = Tracks_z[HT > 1200]
Tracks_fromPV0 = Tracks_fromPV0[HT > 1200]
Tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[HT > 1200]

sph0 = np.zeros(GenParticles_Status.size) #sphericity
sph1 = np.zeros(GenParticles_Status.size) #sphericity
sph2 = np.zeros(GenParticles_Status.size) #sphericity
apl0 = np.zeros(GenParticles_Status.size) #aplanarity
apl1 = np.zeros(GenParticles_Status.size) #aplanarity
apl2 = np.zeros(GenParticles_Status.size) #aplanarity

def makeJets(tracks, R):
    # Cluster AK15 jets
    vectors = np.zeros(tracks.size, np.dtype([('pT', 'f8'), ('eta', 'f8'),
                                              ('phi', 'f8'), ('mass', 'f8')]))
    i = 0
    for track in tracks:
        vectors[i] = np.array((track.pt, track.eta, track.phi, track.mass),
                              np.dtype([('pT', 'f8'), ('eta', 'f8'),
                                        ('phi', 'f8'), ('mass', 'f8')]))
        i += 1
    sequence = pyjet.cluster(vectors, R=R, p=-1)
    jetsAK15 = sequence.inclusive_jets()
    return jetsAK15

def isrTagger(jets):
    mult0 = len(jets[0])
    mult1 = len(jets[1])
    if (mult0 > 130) & (mult1 > 130):
        print("Warning: both multiplicities are above 130!")
    elif (mult0 < 130) & (mult1 < 130):
        print("Warning: both multiplicities are below 130!")
    if mult0 > mult1:
        return uproot_methods.TLorentzVectorArray.from_ptetaphim([jets[1].pt],
                                                                 [jets[1].eta],
                                                                 [jets[1].phi],
                                                                 [jets[1].mass])
    else:
        return uproot_methods.TLorentzVectorArray.from_ptetaphim([jets[0].pt],
                                                                 [jets[0].eta],
                                                                 [jets[0].phi],
                                                                 [jets[0].mass])

for ievt in range(GenParticles_Status.size):
    # Get the particles of ievt event
    genParticles_pt = GenParticles_pt[ievt]
    genParticles_phi = GenParticles_phi[ievt]
    genParticles_eta = GenParticles_eta[ievt]
    genParticles_E = GenParticles_E[ievt]
    genParticles = uproot_methods.TLorentzVectorArray.from_ptetaphie(genParticles_pt,
                                                                     genParticles_eta,
                                                                     genParticles_phi,
                                                                     genParticles_E)
    genParticles_ParentId = GenParticles_ParentId[ievt]
    genParticles_PdgId = GenParticles_PdgId[ievt]
    genParticles_Status = GenParticles_Status[ievt]

    # Tracks in the event
    tracks_x = Tracks_x[ievt]
    tracks_y = Tracks_y[ievt]
    tracks_z = Tracks_z[ievt]
    tracks_E = np.sqrt(tracks_x**2+tracks_y**2+tracks_z**2+0.13957**2)
    tracks = uproot_methods.TLorentzVectorArray.from_cartesian(tracks_x, tracks_y, tracks_z, tracks_E)
    tracks_fromPV0 = Tracks_fromPV0[ievt]
    tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[ievt]

    # Select good tracks
    tracks = tracks[(tracks.pt > 1.) &
                    (tracks.eta < 2.5) &
                    (tracks_fromPV0 >= 2) &
                    (tracks_matchedToPFCandidate > 0)]

    jetsAK15 = makeJets(tracks, 1.5)
    isrJet = isrTagger(jetsAK15)

    # The last copy of the scalar mediator
    scalarParticle = genParticles[(genParticles_PdgId == 25) & (genParticles_Status == 62)]

    # Define mask arrays to select the desired particles
    finalParticles = (genParticles_Status == 1) & (genParticles.pt > 1) & (abs(genParticles.eta) < 3)
    genParticles = genParticles[finalParticles]

    # Boost everything to scalar's rest frame
    genParticles_boosted1 = genParticles.boost(-scalarParticle.p3/scalarParticle.energy)
    genParticles_boosted2 = genParticles.boost(-isrJet.p3/isrJet.energy)

    s0 = eventShapesUtilities.sphericityTensor(genParticles)
    s1 = eventShapesUtilities.sphericityTensor(genParticles_boosted1)
    s2 = eventShapesUtilities.sphericityTensor(genParticles_boosted2)

    sph0[ievt] = eventShapesUtilities.sphericity(s0)
    sph1[ievt] = eventShapesUtilities.sphericity(s1)
    sph2[ievt] = eventShapesUtilities.sphericity(s2)

    apl0[ievt] = eventShapesUtilities.aplanarity(s0)
    apl1[ievt] = eventShapesUtilities.aplanarity(s1)
    apl2[ievt] = eventShapesUtilities.aplanarity(s2)

# Plot results
fig = plt.figure(figsize=(8,8))
ax = plt.gca()

#ax.hist(sph0, 25, histtype='step', label='not boosted', color='b')
#ax.hist(sph1, 25, histtype='step', label='boosted using scalar', color='r')
#ax.hist(sph2, 25, histtype='step', label='boosted using ISR jet', color='g')
#ax.set_xlabel('sphericity', fontsize=18)

ax.hist(apl0, 25, histtype='step', label='not boosted', color='b')
ax.hist(apl1, 25, histtype='step', label='boosted using scalar', color='r')
ax.hist(apl2, 25, histtype='step', label='boosted using ISR jet', color='g')
ax.set_xlabel('aplanarity', fontsize=18)

plt.legend()
plt.show()
