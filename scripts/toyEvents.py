import uproot_methods
import numpy as np
import eventShapesUtilities
import suepsUtilities
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

def particleGeneratorOneBody():
    # One particle at +z
    particles_x = np.array([  0])
    particles_y = np.array([  0])
    particles_z = np.array([100])
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorTwoBodyOpposite():
    # Two identical particles at +/-z
    particles_x = np.array([  0,    0])
    particles_y = np.array([  0,    0])
    particles_z = np.array([100, -100])
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorTwoBodyPerp():
    # Two identical particles at +y,+z
    particles_x = np.array([  0,   0])
    particles_y = np.array([  0, 100])
    particles_z = np.array([100,   0])
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorThreeBody():
    # Three identical particles at +x,+y,+z
    particles_x = np.array([  0,   0, 100,])
    particles_y = np.array([  0, 100,   0,])
    particles_z = np.array([100,   0,   0,])
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorSixBody():
    # Six identical particles at +/-x,+/-y,+/-z
    particles_x = np.array([  0,   0, 100, -100,    0,    0])
    particles_y = np.array([  0, 100,   0,    0, -100,    0])
    particles_z = np.array([100,   0,   0,    0,    0, -100])
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorUniformSphere(N=100):
    # N particles uniformly distributed on a sphere with r = 100
    particles_theta = 2*np.pi*np.random.rand(N)
    particles_u = 2*np.random.rand(N) - 1
    particles_x = 100*np.sqrt(1-particles_u**2)*np.cos(particles_theta)
    particles_y = 100*np.sqrt(1-particles_u**2)*np.sin(particles_theta)
    particles_z = 100*particles_u
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def particleGeneratorUniformCylinder(N=100):
    # N particles uniformly distributed on a sphere with r = 100
    particles_theta = 2*np.pi*np.random.rand(N)
    particles_u = 2*np.random.rand(N) - 1
    particles_x = 100*np.cos(particles_theta)
    particles_y = 100*np.sin(particles_theta)
    particles_z = 100*particles_u
    particles_E = np.sqrt(particles_x**2+particles_y**2+particles_z**2+0.13957**2)
    particles = uproot_methods.TLorentzVectorArray.from_cartesian(particles_x, particles_y, particles_z, particles_E)
    return particles

def calculateEventShapes(method, printOn=False, N=100):
    if method == 'uniformSphere':
        particles = particleGeneratorUniformSphere(N=N)
    elif method == 'uniformCylinder':
        particles = particleGeneratorUniformCylinder(N=N)
    elif method == 'oneBody':
        particles = particleGeneratorOneBody()
    elif method == 'twoBodyOpp':
        particles = particleGeneratorTwoBodyOpposite()
    elif method == 'twoBodyPerp':
        particles = particleGeneratorTwoBodyPerp()
    elif method == 'threeBody':
        particles = particleGeneratorThreeBody()
    elif method == 'sixBody':
        particles = particleGeneratorSixBody()
    else:
        print("Error: method '%s' not recognised"%method)
        return
    s = eventShapesUtilities.sphericityTensor(particles)
    sphericity = eventShapesUtilities.sphericity(s)
    aplanarity = eventShapesUtilities.aplanarity(s)
    C = eventShapesUtilities.C(s)
    D = eventShapesUtilities.D(s)
    circularity = eventShapesUtilities.circularity(particles)
    isotropy = eventShapesUtilities.isotropy(particles)
    if (printOn):
        print('The sphericity tensor is:')
        print(s)
        print('sphericity = %f\naplanarity = %f\nC = %f\nD = %f'
              '\ncircularity = %f\nisotropy = %f\n'%(sphericity, aplanarity, C,
                                                     D, circularity, isotropy))
    return np.array([sphericity, aplanarity, C, D, circularity, isotropy])

events1 = 10000
sph1 = np.zeros(events1)
for i in range(events1):
    evtShapes = calculateEventShapes(method='uniformSphere', N=2)
    sph1[i] = evtShapes[0]

events2 = 1000
sph2 = np.zeros(events2)
for i in range(events2):
    evtShapes = calculateEventShapes(method='uniformSphere', N=100)
    sph2[i] = evtShapes[0]

fig = plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, 700)
ax.set_xlabel('sphericity')
ax.set_ylabel('a.u.')
plt.hist(sph1, bins=25, weights=1*np.ones(events1), histtype='step', label='2 particles - uniformly distributed', color='b')
plt.hist(sph2, bins=20, weights=4*np.ones(events2), histtype='step', label='100 particles - uniformly distributed', color='r')
plt.legend()
plt.show()
