# HostSub_GP
Modeling the 2d spectrum of host galaxies with Gaussian Process for better background subtraction in supernova spectroscopy.

## Input
### Required
- A reduced 2d spectrum (2d array: spatial v.s. spectral)
    - Flat/bias/dark fields calibrated
    - Wavelength calibrated
    - Cosmic ray removed
- Spatial location of the SN
- Seeing
- Spectral resolution
- Instrumental sensitivity function

### Optional
- Slit position/orientation
- Images of the host galaxy (in multiple filters)
- A spectrum of the host galaxy

## Output
- Fluxed calibrated spectra (2d/1d) of the host galaxy/SN 
- Estimated flux uncertainties

## Model
The observed counts at pixel $x$ and wavelength $\lambda$ is composed of
$$
C_\mathrm{obs}(x, \lambda) = \left[f_\mathrm{SN}(x, \lambda) + f_\mathrm{Host}(x, \lambda)\right]\cdot t(\lambda) + f_\mathrm{Sky}(\lambda) + \sigma_C,
$$
where $t(\lambda)$ is the throughput function, and the random noise $\sigma_f\sim N(0, \sigma)$.

The contribution from the SN is modeled as
$$
f_\mathrm{SN}(x, \lambda) = F_\mathrm{SN}(\lambda)\cdot PSF(x,\lambda),
$$
where $F_\mathrm{SN}$ is the 1D spectrum we would like to extract, and $PSF(x, \lambda)$ is the point spread function, which, in principle, is also a function of $\lambda$. The PSF is normalized, i.e., $\int PSF(x,\lambda)\mathrm dx=1$. In many cases we can ignore the wavelength dependence, which is not expected to be huge.

The contribution from the host galaxy is modeled as
$$
f_\mathrm{Host}(x, \lambda) = F_\mathrm{Host}(\lambda)\cdot p_\mathrm{Host}(x, \lambda),
$$
where $F_\mathrm{Host}(\lambda)$ is the 1D spectrum of the host, and $p_\mathrm{Host}(x,\lambda)$ is the spatial profile. By definition, $p_\mathrm{Host}(x,\lambda)$ is also normalized, $\int p_\mathrm{Host}(x,\lambda)\mathrm dx=1$.

We have constraints on $p_\mathrm{Host}$ from multi-band images of the host galaxy,
$$
\int F_\mathrm{Host}(x,\lambda)\cdot p_\mathrm{Host}(x,\lambda)\cdot\lambda t_\mathrm{flt}(\lambda)\mathrm d\lambda = F_\mathrm{Host}(\mathrm{flt})\cdot p_\mathrm{Host}(x;\mathrm{flt}),
$$
where $F_\mathrm{Host}(\mathrm{flt})$ is the broad-band flux of the galaxy in the slit, and $p_\mathrm{Host}(x;\mathrm{flt})$ is the broad-band flux profile. Both can be measured from images.

Ideally, i.e., when flat, tilt, bias are all well calibrated, the sky emission lines will only be a function of $\lambda$. Here we adopt this assumption. If the slit covers $N_\mathrm{pix}$ pixels, then $F_\mathrm{Sky}(\lambda) = N_\mathrm{pix}f_\mathrm{Sky}(\lambda)$.

All that being said, the model looks like
$$
C_\mathrm{obs}(x, \lambda) = \left[F_\mathrm{SN}(\lambda)\cdot PSF(x) + F_\mathrm{Host}(\lambda)\cdot p_\mathrm{Host}(x, \lambda)\right]\cdot t(\lambda) + F_\mathrm{Sky}(\lambda)/N_\mathrm{pix} + \sigma_C.
$$

And when $|x|\ge x_0\gg$ seeing, contribution from the SN is negligible, thus
$$
C'(\lambda;x_0)\equiv\sum_{|x|>x_0}C_\mathrm{obs}(x,\lambda)\simeq F_\mathrm{Host}(\lambda)\cdot t(\lambda)\cdot\sum_{|x|>x_0}p_\mathrm{Host}(x,\lambda) + F_\mathrm{Sky}(\lambda)\cdot \sum_{|x|>x_0}1/N_\mathrm{pix},
$$
which we will model with a 1D Gaussian process,
$$
C'(\lambda;x_0)\sim \mathcal{GP}(0, K_1(\lambda,\lambda^*)).
$$

Independently, the normalized counts at each $\lambda$ is
$$
C''(x,\lambda;x_0)\equiv \frac{C_\mathrm{obs}(x,\lambda)}{C'(\lambda;x_0)}\sim \mathcal{GP}\left(\frac{\bar p_\mathrm{Host}}{\sum_{|x|>x_0}\bar p_\mathrm{Host}}, K_2((x,\lambda),(x^*,\lambda^*))\right).
$$