import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import mplhep as hep
import pyjet
import eventShapesUtilities
import suepsUtilities

plt.style.use(hep.style.ROOT)
boost = False

variable = 'sphericity'
#variable = 'aplanarity'
#variable = 'C'
#variable = 'D'
#variable = 'circularity'
#variable = 'isotropy'

isSignal = True

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

Tracks_E = np.sqrt(Tracks_x**2+Tracks_y**2+Tracks_z**2+0.13957**2)
Tracks = uproot_methods.TLorentzVectorArray.from_cartesian(Tracks_x, Tracks_y, Tracks_z, Tracks_E)
# Select good tracks
Tracks = Tracks[(Tracks.pt > 1.) &
                (Tracks.eta < 2.5) &
                (Tracks_fromPV0 >= 2) &
                (Tracks_matchedToPFCandidate > 0)]

evtShape0 = np.zeros(GenParticles_Status.size)
evtShape1 = np.zeros(GenParticles_Status.size)
evtShape2 = np.zeros(GenParticles_Status.size)
evtShape3 = np.zeros(GenParticles_Status.size)
evtShape4 = np.zeros(GenParticles_Status.size)

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
    tracks = Tracks[ievt]

    # Cluster AK15 jets and find ISR jet
    jetsAK15 = suepsUtilities.makeJets(tracks, 1.5)
    isrJet = suepsUtilities.isrTagger(jetsAK15)

    # The last copy of the scalar mediator
    scalarParticle = genParticles[(genParticles_PdgId == 25) & (genParticles_Status == 62)]

    # Define mask arrays to select the desired particles
    finalParticles = (genParticles_Status == 1) & (genParticles.pt > 1) & (abs(genParticles.eta) < 3)
    #finalParticles_minusISR = finalParticles &
                              #(suepsUtilities.deltar(genParticles.eta, genParticles.phi, isrJet.eta, isrJet.phi) > 1.5)
    genParticles = genParticles[finalParticles]

    # Boost everything to scalar's rest frame
    genParticles_boosted1 = genParticles.boost(-scalarParticle.p3/scalarParticle.energy)
    genParticles_boosted2 = genParticles.boost(-isrJet.p3/isrJet.energy)
    tracks_boosted = tracks.boost(-isrJet.p3/isrJet.energy)

    s0 = eventShapesUtilities.sphericityTensor(genParticles)
    s1 = eventShapesUtilities.sphericityTensor(genParticles_boosted1)
    s2 = eventShapesUtilities.sphericityTensor(genParticles_boosted2)
    s3 = eventShapesUtilities.sphericityTensor(tracks)
    s4 = eventShapesUtilities.sphericityTensor(tracks_boosted)

    if variable == "sphericity":
        evtShape0[ievt] = eventShapesUtilities.sphericity(s0)
        evtShape1[ievt] = eventShapesUtilities.sphericity(s1)
        evtShape2[ievt] = eventShapesUtilities.sphericity(s2)
        evtShape3[ievt] = eventShapesUtilities.sphericity(s3)
        evtShape4[ievt] = eventShapesUtilities.sphericity(s4)
    elif variable == "aplanarity":
        evtShape0[ievt] = eventShapesUtilities.aplanarity(s0)
        evtShape1[ievt] = eventShapesUtilities.aplanarity(s1)
        evtShape2[ievt] = eventShapesUtilities.aplanarity(s2)
        evtShape3[ievt] = eventShapesUtilities.aplanarity(s3)
        evtShape4[ievt] = eventShapesUtilities.aplanarity(s4)
    elif variable == "C":
        evtShape0[ievt] = eventShapesUtilities.C(s0)
        evtShape1[ievt] = eventShapesUtilities.C(s1)
        evtShape2[ievt] = eventShapesUtilities.C(s2)
        evtShape3[ievt] = eventShapesUtilities.C(s3)
        evtShape4[ievt] = eventShapesUtilities.C(s4)
    elif variable == "D":
        evtShape0[ievt] = eventShapesUtilities.D(s0)
        evtShape1[ievt] = eventShapesUtilities.D(s1)
        evtShape2[ievt] = eventShapesUtilities.D(s2)
        evtShape3[ievt] = eventShapesUtilities.D(s3)
        evtShape4[ievt] = eventShapesUtilities.D(s4)
    elif variable == "circularity":
        evtShape0[ievt] = eventShapesUtilities.circularity(genParticles)
        evtShape1[ievt] = eventShapesUtilities.circularity(genParticles_boosted1)
        evtShape2[ievt] = eventShapesUtilities.circularity(genParticles_boosted2)
        evtShape3[ievt] = eventShapesUtilities.circularity(tracks)
        evtShape4[ievt] = eventShapesUtilities.circularity(tracks_boosted)
        print("Event %d processed!"%ievt)
    elif variable == "isotropy":
        evtShape0[ievt] = eventShapesUtilities.isotropy(genParticles)
        evtShape1[ievt] = eventShapesUtilities.isotropy(genParticles_boosted1)
        evtShape2[ievt] = eventShapesUtilities.isotropy(genParticles_boosted2)
        evtShape3[ievt] = eventShapesUtilities.isotropy(tracks)
        evtShape4[ievt] = eventShapesUtilities.isotropy(tracks_boosted)
        print("Event %d processed!"%ievt)
    else:
        print("Error: Unknown event shape variable %s"%variable)

# Plot results
fig = plt.figure(figsize=(8,8))
ax = plt.gca()

ax.hist(evtShape0, 25, histtype='step', label='GEN - not boosted', color='b', linestyle='--')
ax.hist(evtShape1, 25, histtype='step', label='GEN - boosted using scalar', color='b', linestyle='dotted')
ax.hist(evtShape2, 25, histtype='step', label='GEN - boosted using ISR jet', color='b')
ax.hist(evtShape3, 25, histtype='step', label='RECO - not boosted', color='r', linestyle='--')
ax.hist(evtShape4, 25, histtype='step', label='RECO - boosted using ISR jet', color='r')
if variable == "sphericity":
    ax.set_xlabel('sphericity')
elif variable == "aplanarity":
    ax.set_xlabel('aplanarity')
elif variable == "C":
    ax.set_xlabel('C')
elif variable == "D":
    ax.set_xlabel('D')
elif variable == "circularity":
    ax.set_xlabel('circularity')
elif variable == "isotropy":
    ax.set_xlabel('isotropy')

plt.legend()
fig.savefig('Results/%s.pdf'%variable)

plt.show()
