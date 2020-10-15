import numpy as np

def sphericityTensor(genParticles):
    s = np.zeros((3,3))
    s[0][0] = genParticles.x.dot(genParticles.x)
    s[0][1] = genParticles.x.dot(genParticles.y)
    s[0][2] = genParticles.x.dot(genParticles.z)
    s[1][0] = genParticles.y.dot(genParticles.x)
    s[1][1] = genParticles.y.dot(genParticles.y)
    s[1][2] = genParticles.y.dot(genParticles.z)
    s[2][0] = genParticles.z.dot(genParticles.x)
    s[2][1] = genParticles.z.dot(genParticles.y)
    s[2][2] = genParticles.z.dot(genParticles.z)

    s = s/genParticles.p.dot(genParticles.p)

    return s

def sphericity(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    sphericity = (s_eigvalues[0]+s_eigvalues[1])*3./2.
    return sphericity

def aplanarity(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    aplanarity = s_eigvalues[0]*3./2.

    return aplanarity
