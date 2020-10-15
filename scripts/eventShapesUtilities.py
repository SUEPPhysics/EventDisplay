import numpy as np

def sphericityTensor(particles):
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
    sphericity = (s_eigvalues[0]+s_eigvalues[1])*3./2.
    return sphericity

def aplanarity(s):
    s_eigvalues, s_eigvectors = np.linalg.eig(s)
    s_eigvalues = np.sort(s_eigvalues)
    aplanarity = s_eigvalues[0]*3./2.
    return aplanarity
