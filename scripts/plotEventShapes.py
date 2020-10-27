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
# Signal parameters
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
# QCD parameters
htBins = ['1000to1500','1500to2000','2000toInf']
xs = [1207, 119.9, 25.24]
base = '/Users/chrispap/'
#base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.2/2018/NTUP/'
datasets = [base +
            'PrivateSamples.SUEP_2018_mMed-%d_mDark-%d_temp-%d_decay-%s'
            '_13TeV-pythia8_n-100_0_RA2AnalysisTree.root'%(mMed, mDark, temp, decayMode),
            base + 'Autumn18.QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
            base + 'Autumn18.QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_RA2AnalysisTree.root',
           ]
rootfile = datasets[0]
fin = uproot.open(rootfile)

# Attach the branches to numpy arrays
tree = fin['TreeMaker2/PreSelection']
def get_branch(branchname):
    return tree[branchname].array()

HT = get_branch('HT')
Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')

Tracks_x = Tracks_x[HT > 1200]
Tracks_y = Tracks_y[HT > 1200]
Tracks_z = Tracks_z[HT > 1200]
Tracks_fromPV0 = Tracks_fromPV0[HT > 1200]
Tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[HT > 1200]
Tracks_E = np.sqrt(Tracks_x**2+Tracks_y**2+Tracks_z**2+0.13957**2)
Tracks = uproot_methods.TLorentzVectorArray.from_cartesian(Tracks_x, Tracks_y, Tracks_z, Tracks_E)

# Select good tracks
Tracks = Tracks[(Tracks.pt > 1.) & (Tracks.eta < 2.5) & (Tracks_fromPV0 >= 2) &
                (Tracks_matchedToPFCandidate > 0)]

evtShape = np.zeros(Tracks_x.size)
evtShape1 = np.zeros(Tracks_x.size)
evtShape2 = np.zeros(Tracks_x.size)
evtShape3 = np.zeros(Tracks_x.size)
evtShape4 = np.zeros(Tracks_x.size)
evtShape5 = np.zeros(Tracks_x.size)

for ievt in range(Tracks_x.size):
    # Tracks in the event
    tracks = Tracks[ievt]

    # Cluster AK15 jets
    jetsAK15 = suepsUtilities.makeJets(tracks, 1.5)
    isrJet = suepsUtilities.isrTagger(jetsAK15)

    # Subtract ISR
    #tracks_minusISR = tracks[tracks.delta_r(isrJet) > 0.4]

    # Boost everything to scalar's rest frame
    tracks_boosted_minusISR = tracks.boost(-isrJet.p3/isrJet.energy)

    tracks_boosted_minusISR_minus1 = suepsUtilities.removeMaxE(tracks_boosted_minusISR, N=5)
    tracks_boosted_minusISR_minus2 = suepsUtilities.removeMaxE(tracks_boosted_minusISR, N=10)
    tracks_boosted_minusISR_minus3 = suepsUtilities.removeMaxE(tracks_boosted_minusISR, N=20)
    tracks_boosted_minusISR_minus4 = suepsUtilities.removeMaxE(tracks_boosted_minusISR, N=40)
    tracks_boosted_minusISR_minus5 = suepsUtilities.removeMaxE(tracks_boosted_minusISR, N=80)

    s = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR)
    s1 = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR_minus1)
    s2 = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR_minus2)
    s3 = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR_minus3)
    s4 = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR_minus4)
    s5 = eventShapesUtilities.sphericityTensor(tracks_boosted_minusISR_minus5)

    if variable == "sphericity":
        evtShape[ievt] = eventShapesUtilities.sphericity(s)
        evtShape1[ievt] = eventShapesUtilities.sphericity(s1)
        evtShape2[ievt] = eventShapesUtilities.sphericity(s2)
        evtShape3[ievt] = eventShapesUtilities.sphericity(s3)
        evtShape4[ievt] = eventShapesUtilities.sphericity(s4)
        evtShape5[ievt] = eventShapesUtilities.sphericity(s5)
    elif variable == "aplanarity":
        evtShape[ievt] = eventShapesUtilities.aplanarity(s)
    elif variable == "C":
        evtShape[ievt] = eventShapesUtilities.C(s)
    elif variable == "D":
        evtShape[ievt] = eventShapesUtilities.D(s)
    elif variable == "circularity":
        evtShape[ievt] = eventShapesUtilities.circularity(tracks_boosted_minusISR)
        if ievt%100:
            print("Event %d processed!"%ievt)
    elif variable == "isotropy":
        evtShape[ievt] = eventShapesUtilities.isotropy(tracks_boosted_minusISR)
        if ievt%100:
            print("Event %d processed!"%ievt)
    else:
        print("Error: Unknown event shape variable %s"%variable)

# Plot results
fig = plt.figure(figsize=(8,8))
ax = plt.gca()

# Set colormap
values = range(6)
jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

colorVal = scalarMap.to_rgba(values[0])
ax.hist(evtShape, bins=25, range=(0, 1), histtype='step', label='minus 0', color=colorVal)
colorVal = scalarMap.to_rgba(values[1])
ax.hist(evtShape1, bins=25, range=(0, 1), histtype='step', label='minus 5', color=colorVal)
colorVal = scalarMap.to_rgba(values[2])
ax.hist(evtShape2, bins=25, range=(0, 1), histtype='step', label='minus 10', color=colorVal)
colorVal = scalarMap.to_rgba(values[3])
ax.hist(evtShape3, bins=25, range=(0, 1), histtype='step', label='minus 20', color=colorVal)
colorVal = scalarMap.to_rgba(values[4])
ax.hist(evtShape4, bins=25, range=(0, 1), histtype='step', label='minus 40', color=colorVal)
colorVal = scalarMap.to_rgba(values[5])
ax.hist(evtShape5, bins=25, range=(0, 1), histtype='step', label='minus 80', color=colorVal)
if variable == "sphericity":
    ax.set_xlabel('sphericity', fontsize=18)
elif variable == "aplanarity":
    ax.set_xlabel('aplanarity', fontsize=18)
elif variable == "C":
    ax.set_xlabel('C', fontsize=18)
elif variable == "D":
    ax.set_xlabel('D', fontsize=18)
elif variable == "circularity":
    ax.set_xlabel('circularity', fontsize=18)
elif variable == "isotropy":
    ax.set_xlabel('isotropy', fontsize=18)

plt.legend()
#fig.savefig('Results/%s.pdf'%variable)

#plt.show()
