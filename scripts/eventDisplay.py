import uproot
import uproot_methods
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyjet


# Input section  -  You may want to edit these
# File selection
mMed = 1000
mDark = 2
temp = 2
#decayMode = 'darkPho'
decayMode = 'darkPhoHad'
base = '/Users/chrispap/'
# xrootd is not working properly in Python3 :(
#base = 'root://cmseos.fnal.gov//store/user/kdipetri/SUEP/Production_v0.1/2018/NTUP/'
datasets = [base +
            'PrivateSamples.SUEP_2018_mMed-%d_mDark-%d_temp-%d_decay-%s'
            '_13TeV-pythia8_n-100_0_RA2AnalysisTree.root'%(mMed, mDark, temp, decayMode),
           ]
# if you want multiple plots then change multi to True
# Warning: multiplot funtion is currently broken
multi = False
if multi:
    figure, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
    axs_flat = axs.flatten()

# Switch to true if you want to boost along the scalar 4-momentum
boost = False
# Switch to true to save the figure as a PDF
save = True
# The event number the loop starts running form
event = 46


# Get the file and import using uproot
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
GenParticles_ParentId = get_branch('GenParticles_ParentId')
GenParticles_PdgId = get_branch('GenParticles_PdgId')
GenParticles_Status = get_branch('GenParticles_Status')
Tracks_x = get_branch('Tracks.fCoordinates.fX')
Tracks_y = get_branch('Tracks.fCoordinates.fY')
Tracks_z = get_branch('Tracks.fCoordinates.fZ')
Tracks_fromPV0 = get_branch('Tracks_fromPV0')
Tracks_matchedToPFCandidate = get_branch('Tracks_matchedToPFCandidate')
JetsAK8_pt = get_branch('JetsAK8.fCoordinates.fPt')
JetsAK8_eta = get_branch('JetsAK8.fCoordinates.fEta')
JetsAK8_phi = get_branch('JetsAK8.fCoordinates.fPhi')
JetsAK8_E = get_branch('JetsAK8.fCoordinates.fE')
HT = get_branch('HT')

def get_dr_ring(dr, phi_c=0, eta_c=0, n_points=100):
    deta = np.linspace(-dr, +dr, n_points)
    dphi = np.sqrt(dr**2 - np.square(deta))
    deta = eta_c+np.concatenate((deta, deta[::-1]))
    dphi = phi_c+np.concatenate((dphi, -dphi[::-1]))
    return dphi, deta

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

# Main plotting function
def plot(ievt, ax=None, boost=False):
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

    # Get tracks information
    tracks_x = Tracks_x[event]
    tracks_y = Tracks_y[event]
    tracks_z = Tracks_z[event]
    tracks_E = np.sqrt(tracks_x**2+tracks_y**2+tracks_z**2+0.13957**2)
    tracks = uproot_methods.TLorentzVectorArray.from_cartesian(tracks_x, tracks_y, tracks_z, tracks_E)
    tracks_fromPV0 = Tracks_fromPV0[event]
    tracks_matchedToPFCandidate = Tracks_matchedToPFCandidate[event]

    # Get the AK8 jets of the event
    jetsAK8_pt = JetsAK8_pt[ievt]
    jetsAK8_eta = JetsAK8_eta[ievt]
    jetsAK8_phi = JetsAK8_phi[ievt]
    jetsAK8_E = JetsAK8_E[ievt]
    jetsAK8 = uproot_methods.TLorentzVectorArray.from_ptetaphie(jetsAK8_pt,
                                                                jetsAK8_eta,
                                                                jetsAK8_phi,
                                                                jetsAK8_E)

    # The last copy of the scalar mediator
    scalarParticle = genParticles[(genParticles_PdgId == 25) & (genParticles_Status == 62)]

    # Define mask arrays to select the desired particles
    finalParticles = (genParticles_Status == 1) & (genParticles.pt > 1)
    fromScalarParticles = genParticles_ParentId == 999998
    isrParticles = genParticles_ParentId != 999998

    # Select good tracks
    tracks = tracks[(tracks.pt > 1.) &
                    (tracks.eta < 2.5) &
                    (tracks_fromPV0 >= 2) &
                    (tracks_matchedToPFCandidate > 0)]
    # and make AK15 jets
    jetsAK15_list = makeJets(tracks, 1.5)
    jetsAK15_pt = np.zeros(len(jetsAK15_list))
    jetsAK15_eta = np.zeros(len(jetsAK15_list))
    jetsAK15_phi = np.zeros(len(jetsAK15_list))
    jetsAK15_m = np.zeros(len(jetsAK15_list))
    i = 0
    for jet in jetsAK15_list:
        jetsAK15_pt[i] = jet.pt
        jetsAK15_eta[i] = jet.eta
        jetsAK15_phi[i] = jet.phi
        jetsAK15_m[i] = jet.mass
        i += 1
    jetsAK15 = uproot_methods.TLorentzVectorArray.from_ptetaphim(jetsAK15_pt,
                                                                 jetsAK15_eta,
                                                                 jetsAK15_phi,
                                                                 jetsAK15_m)
    #jetsAK15 = jetsAK15[jetsAK15.pt > 30]
    #jetsAK15 = jetsAK15[0:2]

    # Apply the selection criteria to get the final particle arrays
    # 10 arrays of final particles in total
    # Dividing to e, mu, gamma, pi, all other hadrons
    # for particles that come from the scalar mediator or not
    fromScalarParticles_e = genParticles[finalParticles &
                                         fromScalarParticles &
                                         (abs(genParticles_PdgId) == 11)]
    fromScalarParticles_mu = genParticles[finalParticles &
                                          fromScalarParticles &
                                          (abs(genParticles_PdgId) == 13)]
    fromScalarParticles_gamma = genParticles[finalParticles &
                                             fromScalarParticles &
                                             (abs(genParticles_PdgId) == 22)]
    fromScalarParticles_pi = genParticles[finalParticles &
                                          fromScalarParticles &
                                          (abs(genParticles_PdgId) == 211)]
    fromScalarParticles_hadron = genParticles[finalParticles &
                                              fromScalarParticles &
                                              (abs(genParticles_PdgId) > 100)]

    isrParticles_e = genParticles[finalParticles & isrParticles &
                                  (abs(genParticles_PdgId) == 11)]
    isrParticles_mu = genParticles[finalParticles & isrParticles &
                                   (abs(genParticles_PdgId) == 13)]
    isrParticles_gamma = genParticles[finalParticles & isrParticles &
                                      (abs(genParticles_PdgId) == 22)]
    isrParticles_pi = genParticles[finalParticles & isrParticles &
                                   (abs(genParticles_PdgId) == 211)]
    isrParticles_hadron = genParticles[finalParticles & isrParticles &
                                       (abs(genParticles_PdgId) > 100)]

    # Boost everything to scalar's rest frame
    if boost == True:
        boost_vector = -scalarParticle.p3/scalarParticle.energy
        fromScalarParticles_e = fromScalarParticles_e.boost(boost_vector)
        fromScalarParticles_mu = fromScalarParticles_mu.boost(boost_vector)
        fromScalarParticles_gamma = fromScalarParticles_gamma.boost(boost_vector)
        fromScalarParticles_pi = fromScalarParticles_pi.boost(boost_vector)
        fromScalarParticles_hadron = fromScalarParticles_hadron.boost(boost_vector)
        isrParticles_e = isrParticles_e.boost(boost_vector)
        isrParticles_mu = isrParticles_mu.boost(boost_vector)
        isrParticles_gamma = isrParticles_gamma.boost(boost_vector)
        isrParticles_pi = isrParticles_pi.boost(boost_vector)
        isrParticles_hadron = isrParticles_hadron.boost(boost_vector)
        jetsAK8 = jetsAK8.boost(boost_vector)
        scalarParticle = scalarParticle.boost(boost_vector)

    # Initialize plotting
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    # Plot parameters
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r'$\phi$', fontsize=18)
    ax.set_ylabel(r'$\eta$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Function that sets the scaling for markers
    # Two methods for the moment: use tanh or square.
    # Scaling using just the energy is also an option
    def scale(particles, scalar, method=3):
        """Just to scale to a reasonable dot size"""
        energies = particles.energy
        e_max = scalar.energy
        if len(energies) == 0: return []
        if method == 1:
            e_normed = 500000.*np.square(energies/e_max)
        elif method == 2:
            e_normed = 1000.*np.tanh(energies/e_max)
        else:
            e_normed = 2500.*energies/e_max
        return e_normed

    # Add scatters to figure
    ax.scatter(fromScalarParticles_e.phi, fromScalarParticles_e.eta,
               s=scale(fromScalarParticles_e,scalarParticle),
               c='xkcd:blue', marker='o')
    ax.scatter(fromScalarParticles_mu.phi, fromScalarParticles_mu.eta,
               s=scale(fromScalarParticles_mu,scalarParticle),
               c='xkcd:blue', marker='v')
    ax.scatter(fromScalarParticles_gamma.phi, fromScalarParticles_gamma.eta,
               s=scale(fromScalarParticles_gamma,scalarParticle),
               c='xkcd:blue', marker='s')
    ax.scatter(fromScalarParticles_pi.phi, fromScalarParticles_pi.eta,
               s=scale(fromScalarParticles_pi,scalarParticle),
               c='xkcd:blue', marker='P')
    ax.scatter(fromScalarParticles_hadron.phi, fromScalarParticles_hadron.eta,
               s=scale(fromScalarParticles_hadron,scalarParticle),
               c='xkcd:blue', marker='*')
    ax.scatter(isrParticles_e.phi, isrParticles_e.eta,
               s=scale(isrParticles_e,scalarParticle),
               c='xkcd:magenta', marker='o')
    ax.scatter(isrParticles_mu.phi, isrParticles_mu.eta,
               s=scale(isrParticles_mu,scalarParticle),
               c='xkcd:magenta', marker='v')
    ax.scatter(isrParticles_gamma.phi, isrParticles_gamma.eta,
               s=scale(isrParticles_gamma,scalarParticle),
               c='xkcd:magenta', marker='s')
    ax.scatter(isrParticles_pi.phi, isrParticles_pi.eta,
               s=scale(isrParticles_pi,scalarParticle),
               c='xkcd:magenta', marker='P')
    ax.scatter(isrParticles_hadron.phi, isrParticles_hadron.eta,
               s=scale(isrParticles_hadron,scalarParticle),
               c='xkcd:magenta', marker='*')

    # Add the scalar mediator to the plot
    ax.scatter(scalarParticle.phi, scalarParticle.eta,
               s=scale(scalarParticle,scalarParticle), facecolors='none',
               edgecolors='xkcd:red')

    # Add AK8 and AK15 jets to the plot
    ax.scatter(jetsAK8.phi, jetsAK8.eta, s=scale(jetsAK8,scalarParticle),
               facecolors='none', edgecolors='xkcd:bright green')
    #ax.scatter(jetsAK15.phi, jetsAK15.eta, s=scale(jetsAK15,scalarParticle),
               #facecolors='none', edgecolors='xkcd:bright yellow')
    for jet in jetsAK8:
        phis, etas = get_dr_ring(0.8, jet.phi, jet.eta)
        ax.plot(phis, etas, color='xkcd:bright green', linestyle='--')
        print("Jet: pT=%d, eta=%.2f, phi=%.2f\n"%(jet.pt, jet.eta, jet.phi))
    #for jet in jetsAK15:
        #phis, etas = get_dr_ring(1.5, jet.phi, jet.eta)
        #ax.plot(phis, etas, color='xkcd:bright yellow', linestyle='--')


    # Legend 1 is particle type
    line1 = ax.scatter([-100], [-100], label='$e$',marker='o', c='xkcd:black')
    line2 = ax.scatter([-100], [-100], label='$\mu$', marker='v', c='xkcd:black')
    line3 = ax.scatter([-100], [-100], label='$\gamma$', marker='s', c='xkcd:black')
    line4 = ax.scatter([-100], [-100], label='$\pi$', marker='P', c='xkcd:black')
    line5 = ax.scatter([-100], [-100], label='other hadron', marker='*', c='xkcd:black')
    line6 = ax.scatter([-100], [-100], label='Scalar mediator', marker='o',
                       facecolors='none', edgecolors='xkcd:red')
    line7 = ax.scatter([-100], [-100], label='AK8 jets', marker='o',
                       facecolors='none', edgecolors='xkcd:bright green')
    line8 = ax.scatter([-100], [-100], label='AK15 jets', marker='o',
                       facecolors='none', edgecolors='xkcd:bright yellow')
    first_legend = plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7],
                              loc='upper right', fontsize=12)
    ax.add_artist(first_legend)

    # Legend 2 is about particle origin
    blue_patch = mpatches.Patch(color='xkcd:blue', label='from scalar')
    magenta_patch = mpatches.Patch(color='xkcd:magenta', label='not from scalar')
    plt.legend(handles=[blue_patch, magenta_patch],loc='upper left')

    # build a rectangle in axes coords
    left, width = .0, 1.
    bottom, height = .0, 1.
    center = left + width/2.
    right = left + width
    top = bottom + height

    # axes coordinates are 0,0 is bottom left and 1,1 is upper right
    p = mpatches.Rectangle((left, bottom), width, height,
        fill=False, transform=ax.transAxes, clip_on=False)

    ax.add_patch(p)
    # Print event number
    ax.text(left, top, 'Event %d'%ievt, horizontalalignment='left',
            verticalalignment='bottom', transform=ax.transAxes, fontsize=12)
    # Print sample details
    ax.text(right, top, 'mMed=%d$\,$GeV,mDark=%d$\,$GeV,T=%d$\,$K,'
            '%s'%(mMed,mDark,temp,decayMode), horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes, fontsize=12)
    # Print details about cuts
    ax.text(left+0.02, bottom+0.01, 'Final particles have $P_{T}>1\,$GeV',
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=12)
    # Print details of scalar mediator
    ax.text(left+0.02, bottom+0.05, 'Scalar mediator $P_{T}=%d\,$GeV'%(scalarParticle.pt),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=12)

    if (boost == False) & (save == True):
        fig.savefig('Results/mMed%d_mDark%d_temp%d_decay-%s_'
                    'Event%d.pdf'%(mMed, mDark, temp, decayMode,event))
    elif (boost == True) & (save == True):
        fig.savefig('Results/mMed%d_mDark%d_temp%d_decay-%s_'
                    'Event%d_boosted.pdf'%(mMed, mDark, temp, decayMode,event))


# The program runs through this loop
j = 0
for i in range(event,100+event):
    if HT[i] < 1200: continue
    if multi == False:
        plot(i,boost=boost)
        break;
    else:
        plot(i,ax=axs_flat[j])
        j += 1

plt.show();
