# HostSub_GP
Modeling the 2d spectrum of host galaxies with Gaussian process (GP) for better background subtraction in supernova spectroscopy.

## Input
### Required
- A reduced 2d spectrum (2d array: spatial v.s. spectral)
    - Flat/bias/dark fields calibrated
    - Wavelength calibrated
    - Cosmic ray removed
- Spatial location of the SN
- Seeing
- Spectral resolution
- Slit position/orientation
- Images of the host galaxy (in multiple filters)

### Optional (TODO)
- Instrumental sensitivity function
- A spectrum of the host galaxy

## Output
- Sky/host background subtracted 2D spectra

## Model
The observed counts at pixel $x$ and wavelength $\lambda$ is composed of

$$C_\mathrm{Obs}(x, \lambda) = f_\mathrm{SN}(x, \lambda) + f_\mathrm{Host}(x, \lambda)  + f_\mathrm{Sky}(\lambda) + \sigma_C,$$

where the random noise $\sigma_f\sim N(0, \sigma)$.

The contribution from the SN is modeled as

$$f_\mathrm{SN}(x, \lambda) = F_\mathrm{SN}(\lambda)\cdot PSF(x,\lambda),$$

where $F_\mathrm{SN}$ is the 1D spectrum we would like to extract, and $PSF(x, \lambda)$ is the point spread function, which, in principle, is also a function of $\lambda$. The PSF is normalized, i.e., $\int PSF(x,\lambda)\mathrm dx=1$. In many cases we can ignore the wavelength dependence, which is not expected to be huge.

The contribution from the host galaxy is modeled as

$$f_\mathrm{Host}(x, \lambda) = F_\mathrm{Host}(\lambda)\cdot \xi(x, \lambda),$$

where $F_\mathrm{Host}(\lambda)$ is the 1D spectrum of the host, and $\xi(x,\lambda)$ is the spatial profile, which is also normalized, 

$$\int_{|x|>x_M} \xi(x,\lambda)\mathrm dx=1,$$

where $x_M$ stands for the radius of the aperture where the SN is masked. This typically requires $x_M\gg$ seeing.

Ideally, i.e., when flat, tilt, bias are all well calibrated, the sky emission lines will only be a function of $\lambda$. Here we adopt this assumption.

All that being said, the model looks like

$$C_\mathrm{Obs}(x, \lambda) = F_\mathrm{SN}(\lambda)\cdot PSF(x) + F_\mathrm{Host}(\lambda)\cdot \xi(x, \lambda) + f_\mathrm{Sky}(\lambda) + \sigma_C.$$

To get rid of the sky background, we estimate the mean counts outside some $x_G > x_M\gg$ seeing, i.e., the global background

$$C_G(\lambda) = \langle C_\mathrm{Obs}(x,\lambda)\rangle_{|x|>x_G} = F_\mathrm{Host}(\lambda) \langle \xi(x,\lambda)\rangle_{|x|>x_G} + f_\mathrm{Sky}(\lambda).$$

By subtracting the global background, we remove the sky emission and get

$$\widetilde C(x,\lambda) \equiv C_\mathrm{Obs}(x, \lambda) - C_G(\lambda) = F_\mathrm{SN}(\lambda)\cdot PSF(x) + F_\mathrm{Host}(\lambda)\left[\xi(x,\lambda) - \langle \xi(x,\lambda)\rangle_{|x|>x_G}\right].$$

And when $|x|\ge x_M\gg$ seeing, contribution from the SN is negligible, thus

$$\widetilde C_{\mathrm{1D}}(\lambda)\equiv\int_{|x|>x_M}\widetilde C(x,\lambda)\mathrm dx\simeq F_\mathrm{Host}(\lambda) \cdot\left[\left({l_\mathrm{Slit} - l_\mathrm{Mask}}\right)\left(\langle \xi(x,\lambda)\rangle_{|x|>x_M} - \langle \xi(x,\lambda)\rangle_{|x|>x_G}\right)\right],$$

which we will model with a 1D GP

$$\widetilde C_{\mathrm{1D}}(\lambda)\sim \mathcal{GP}(\mu_\mathrm{1D}, K_1(\lambda,\lambda^\ast; l_\mathrm{1D})),$$

conditioned on the integrated observed flux $\hat C_{\mathrm{1D}}$. Here $\mu_\mathrm{1D}$ is a fixed mean value of the GP, $l_\mathrm{1D}$ is the scaling factor of the kernel, which should be the order of the spectral resolution.

Independently, the normalized counts at each $\lambda$ is

$$\widetilde C_{\mathrm{2D}}(x,\lambda)\equiv \frac{\widetilde C(x,\lambda)}{\widetilde C_\mathrm{1D}(\lambda)} = \frac{\xi(x,\lambda)-\langle \xi(x,\lambda)\rangle_{|x|>x_G}}{\langle \xi(x,\lambda)\rangle_{|x|>x_M} - \langle \xi(x,\lambda)\rangle_{|x|>x_G}}\frac{1}{l_\mathrm{Slit} - l_\mathrm{Mask}}.$$

We will model it with a 2D GP

$$\widetilde C_{\mathrm{2D}}(x,\lambda)\sim\mathcal{GP}(\mu_\mathrm{2D}(x,\lambda), K_2((x,\lambda),(x^\ast,\lambda^\ast); l_\mathrm{2D})).$$

The mean function $\mu_\mathrm{2D}(x,\lambda)$ is estimated from the (multi-band) images of the galaxy within the slit, and $l_\mathrm{2D}$ is the 2D scaling factor. Note that we expect the scaling factor on the spectral orientation is much greater than $l_\mathrm{1D}$, which is essentially why we would like to separate the two independent GPs.