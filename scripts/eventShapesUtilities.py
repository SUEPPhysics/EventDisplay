import numpy as np
import math

## Check if this is the correct dot product
def sphericityTensor(particles,r=2):
    s = np.zeros((3,3))
    s[0][0] = particles.x.dot(particles.x)
    s[0][1] = particles.x.dot(particles.y)
    s[0][2] = particles.x.dot(particles.z)
    s[1][0] = particles.y.dot(particles.x)
    s[1][1] = particles.y.dot(particles.y)
    s[1][2] = particles.y.dot(particles.z)
    s[2][0] = particles.z.dot(particles.x)
    s[2][1] = particles.z.dot(particles.y)
    s[2][2] = particles.z.dot(particles.z)
    s = s/particles.p.dot(particles.p)
    return s

def sphericity(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    sphericity = 1.5*(s_eigvalues[0]+s_eigvalues[1])
    return sphericity

def aplanarity(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    aplanarity = 1.5*s_eigvalues[0]
    return aplanarity

def C(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    C = 3.*(s_eigvalues[2]*s_eigvalues[0]+
            s_eigvalues[2]*s_eigvalues[1]+
            s_eigvalues[0]*s_eigvalues[1])
    return C

def D(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    D = 27.*(s_eigvalues[0]*s_eigvalues[1]*s_eigvalues[2])
    return D

# In fact, this is the linearized circularity
# (this is the definition also used in CMSSW)
def circularity(particles, numberOfSteps=100):
    phi = np.linspace(0,2*math.pi,numberOfSteps)
    pTsum = np.sum(particles.pt)
    nTs = np.array([np.cos(phi), np.sin(phi)])
    nTs = nTs.transpose()
    pTcomponents = np.array([particles.p3.x, particles.p3.y])
    pTcomponents = pTcomponents.transpose()
    sum_pn = np.zeros(numberOfSteps)
    i = 0
    for nT in nTs:
        for pTcomponent in pTcomponents:
            sum_pn[i] += np.abs(nT.dot(pTcomponent))
        i+=1
    return math.pi*np.amin(sum_pn)/(2*pTsum)

def isotropy(particles, numberOfSteps=100):
    phi = np.linspace(0,2*math.pi,numberOfSteps)
    nTs = np.array([np.cos(phi), np.sin(phi)])
    nTs = nTs.transpose()
    pTcomponents = np.array([particles.p3.x, particles.p3.y])
    pTcomponents = pTcomponents.transpose()
    sum_pn = np.zeros(numberOfSteps)
    i = 0
    for nT in nTs:
        for pTcomponent in pTcomponents:
            sum_pn[i] += np.abs(nT.dot(pTcomponent))
        i+=1
    eMin = np.amin(sum_pn)
    eMax = np.amax(sum_pn)
    return (eMax-eMin)/eMax
