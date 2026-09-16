"""IMRPhenomXHM parameter-space calibration fits in JAX.

Inherited coefficient expressions attributed to the author ancillary release for
Garcia-Quiros et al. (2020), arXiv:2001.10914v2.
"""

from __future__ import annotations

import jax.numpy as jnp


def _nonnegative_sqrt(value: jnp.ndarray) -> jnp.ndarray:
    """Exact sqrt on positive inputs, with zero value/subgradient at the boundary.

    The unselected sqrt branch is evaluated at one, preventing the 0*inf
    backward operation at equal masses. No positive-value cutoff is introduced.
    """
    positive = value > 0.0
    root = jnp.sqrt(jnp.where(positive, value, 1.0))
    return jnp.where(positive, root, 0.0)


class XHMParameterFits:
    """Calibrated rational and polynomial fits for IMRPhenomXHM higher modes in JAX."""

    @staticmethod
    def compute_spin_variables(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray):
        """Compute spin variables S_tot and delta."""
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        delta = _nonnegative_sqrt(diff)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        den = jnp.maximum(1.0 - 2.0 * eta, 1e-12)
        S = (m1**2 * chi1 + m2**2 * chi2) / den
        return delta, m1, m2, S

    # -------------------------------------------------------------------------
    # Quasi-Normal Modes (QNMs)
    # -------------------------------------------------------------------------
    @staticmethod
    def qnm_fring_21(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fring for mode (2,1) as rational function of final spin chi."""
        return (0.05947169566573467 - 0.07587273723102551*chi +     0.021980332978392875*chi**2 - 0.0018961876378928344*chi**3 +     0.0011658539452450002*chi**4 - 0.0003384978210166078*chi**5)/   (1 - 1.4440542001828458*chi + 0.49273151019467565*chi**2)

    @staticmethod
    def qnm_fring_22(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fring for mode (2,2) as rational function of final spin chi."""
        return (0.05947169566573468 - 0.14989793140040952*chi +     0.09535636998827782*chi**2 + 0.022609338033479816*chi**3 -     0.025017210261448576*chi**4 - 0.005852492572196134*chi**5 +     0.0027489358282817337*chi**6 + 0.0005822121915782328*chi**7)/   (1 - 2.857016345962658*chi + 2.373341833103157*chi**2 -     0.6037001154822206*chi**4 + 0.08738073298206948*chi**6)

    @staticmethod
    def qnm_fring_32(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fring for mode (3,2) as rational function of final spin chi."""
        return (0.0954043624521206 - 0.13720480688379635*chi +     0.031218647847282344*chi**2 - 0.0004897788407119615*chi**3 +     0.011464267778238106*chi**4 + 0.001999218087875153*chi**5 -     0.001493094147046422*chi**6)/(1 - 1.6628470814705156*chi +     0.5773688193561025*chi**2 + 0.11771131121402911*chi**4 -     0.026740614067243006*chi**6)

    @staticmethod
    def qnm_fring_33(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fring for mode (3,3) as rational function of final spin chi."""
        return (0.09540436245212058 - 0.22780534714220443*chi +     0.13377437287667318*chi**2 + 0.033361963010878265*chi**3 -     0.030709971579549696*chi**4 - 0.006723622942584775*chi**5 +     0.002699789485014166*chi**6)/(1 - 2.724613686365119*chi +     2.140755614298282*chi**2 - 0.468981591372166*chi**4 +     0.052846485272773074*chi**6)

    @staticmethod
    def qnm_fring_44(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fring for mode (4,4) as rational function of final spin chi."""
        return (0.12878343723093433 - 0.22188404754333252*chi +     0.08072346598554622*chi**2 + 0.018228949867392645*chi**3 -     0.004773929513803944*chi**4 - 0.000487752608647949*chi**5)/   (1 - 2.0680862421216375*chi + 1.1603274231645002*chi**2 -     0.09029582520502108*chi**4)

    @staticmethod
    def qnm_fdamp_21(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fdamp for mode (2,1) as rational function of final spin chi."""
        return (0.006122662443153689 - 0.3527329342371466*(-1 + chi) +     2.0691582389564616*(-1 + chi)**2 + 0.06971544107947354*(-1 + chi)**3 -     0.28114850788176*(-1 + chi)**4 + 0.0028814691599800754*(-1 + chi)**5)/   (1 - 32.61993194255531*(-1 + chi) + 131.91477112834812*(-1 + chi)**2 -     19.0335620314733*(-1 + chi)**4)

    @staticmethod
    def qnm_fdamp_22(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fdamp for mode (2,2) as rational function of final spin chi."""
        return (0.014158792290965177 - 0.03698927933438008*chi +     0.026822312040690962*chi**2 + 0.0008490910346153032*chi**3 -     0.00484385665747365*chi**4 - 0.00014745117461456713*chi**5 +     0.0001504132482864047*chi**6)/(1 - 2.59007605887131*chi +     1.8952427080249667*chi**2 - 0.3141568097404366*chi**4 +     0.009000115980058154*chi**6)

    @staticmethod
    def qnm_fdamp_32(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fdamp for mode (3,2) as rational function of final spin chi."""
        return (0.014754148319335945 - 0.03444452267551954*chi +     0.021670131053693586*chi**2 + 0.0014940390781148472*chi**3 -     0.003470184589024902*chi**4)/(1 - 2.3198412558555495*chi +     1.5083840528053443*chi**2 - 0.18755907646053682*chi**4)

    @staticmethod
    def qnm_fdamp_33(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fdamp for mode (3,3) as rational function of final spin chi."""
        return (0.014754148319335945 - 0.027342638172676126*chi +     0.011768218352063733*chi**2 + 0.0008641978105760124*chi**3 +     0.000014272007791441222*chi**4 - 0.00005205518385328013*chi**5)/   (1 - 1.8317887872994476*chi + 0.8347260664295573*chi**2)

    @staticmethod
    def qnm_fdamp_44(chi: jnp.ndarray) -> jnp.ndarray:
        """QNM fdamp for mode (4,4) as rational function of final spin chi."""
        return (0.014986837706077725 - 0.017849784560800525*chi -     0.0008171267159120031*chi**2 + 0.00031927423914265217*chi**3 +     0.002264684298282546*chi**4 + 0.0003011889529114187*chi**5 +     0.000808321125918881*chi**6)/(1 - 1.1766560862080135*chi +     0.006439316812899679*chi**2 + 0.1250160855378052*chi**4 +     0.05161681929353713*chi**6)

    # -------------------------------------------------------------------------
    # Spherical-Spheroidal Mixing Coefficients
    # -------------------------------------------------------------------------
    @staticmethod
    def mixing_alpha222(chi: jnp.ndarray):
        """Complex mixing coefficient alpha222 as (real, imag) tuple for final spin chi."""
        re = (1 - 2.295699213216995*chi + 1.4619885788982425*chi**2 +     0.00432963614839615*chi**3 - 0.16956669291917906*chi**4 -     0.0006267847253053242*chi**5)/(1 - 2.2956976283416806*chi +     1.4646337173895434*chi**2 - 0.1684322162942244*chi**4 -     0.00007150530536478032*chi**6)
        im = (-0.009885910865482132*chi + 0.012279331664776998*chi**2 +     0.001523728817468495*chi**3 - 0.001871754864285381*chi**4 -     0.0014135099238647992*chi**5 - 0.0006340973594701062*chi**6)/   (1 - 1.8037813598881576*chi + 0.7231092044311039*chi**2 +     0.10727169825436109*chi**4 - 0.025834219017477553*chi**6)
        return re, im

    @staticmethod
    def mixing_alpha223(chi: jnp.ndarray):
        """Complex mixing coefficient alpha223 as (real, imag) tuple for final spin chi."""
        re = (0.11258219435129532*chi - 0.23083531810930033*chi**2 +     0.11566878900797303*chi**3 + 0.01845050500432622*chi**4 -     0.013432786600323298*chi**5 - 0.002256017488494162*chi**6)/   (1 - 2.296974390894016*chi + 1.4790189724079015*chi**2 -     0.18718980232565283*chi**4 + 0.006078101838110706*chi**6)
        im = (-0.01741034348700732*chi + 0.042132027730105805*chi**2 -     0.02901904283485293*chi**3 + 0.0006022057021906296*chi**4 +     0.004037053170210619*chi**5 - 0.0003435083693841253*chi**6)/   (1 - 2.205857911577023*chi + 1.338313134042712*chi**2 -     0.13215733098064958*chi**4 + 0.0003498284142573707*chi**6)
        return re, im

    @staticmethod
    def mixing_alpha322(chi: jnp.ndarray):
        """Complex mixing coefficient alpha322 as (real, imag) tuple for final spin chi."""
        re = (-0.07019334024581896*chi + 0.13320603013912857*chi**2 -     0.05680079265436778*chi**3 - 0.012639575688787628*chi**4 +     0.005348058369391428*chi**5 + 0.0009687270382732074*chi**6)/   (1 - 2.2412418374321113*chi + 1.3896075701394461*chi**2 -     0.1477336567452966*chi**4)
        im = (0.016707359440555486*chi - 0.02582193201328815*chi**2 +     0.005887618582476872*chi**3 + 0.0005218228170635253*chi**4 +     0.0024659764782449666*chi**5 + 0.0002432193122631298*chi**6)/   (1 - 1.5852676885553714*chi + 0.4463453474394013*chi**2 +     0.14633534716145832*chi**4 - 0.005820380876632491*chi**6)
        return re, im

    @staticmethod
    def mixing_alpha333(chi: jnp.ndarray):
        """Complex mixing coefficient alpha333 as (real, imag) tuple for final spin chi."""
        re = (1 - 2.107851868107966*chi + 1.1906389430053965*chi**2 +     0.022448480368943948*chi**3 - 0.09593472625375113*chi**4 -     0.0021343356572759114*chi**5 - 0.0053193794464599636*chi**6)/   (1 - 2.107851031248286*chi + 1.2043480484290372*chi**2 -     0.0891021664149287*chi**4 - 0.005471615908522895*chi**6)
        im = (0.03605199432992765*chi - 0.08508249902901473*chi**2 +     0.05285289288410341*chi**3 + 0.005588124106260517*chi**4 -     0.009144706660857042*chi**5 - 0.0002636162163554519*chi**6)/   (1 - 2.359959805984371*chi + 1.5581575366642713*chi**2 -     0.20080528931110736*chi**4 + 0.002894115171190354*chi**6)
        return re, im

    # -------------------------------------------------------------------------
    # Amplitude Collocation Point Fits
    # -------------------------------------------------------------------------
    @staticmethod
    def amp_insp1_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp1 for mode (2,1)."""
        return _nonnegative_sqrt(1 - 4*eta)*(0.037868557189995156 + 0.10740090317702103*eta +         1.963812986867654*eta**2 - 16.706455229589558*eta**3 +         69.75910808095745*eta**4 - 98.3062466823662*eta**5) +       _nonnegative_sqrt(1 - 4*eta)*(S**2*(-0.008044970210401221 - 0.5985436493302652*           eta**2) + S*(-0.007963757232702217 + 0.10627108779259971*eta -           0.4735861262934256*eta**2) +         S**3*(0. - 0.08217216660522084*eta**2)) - 0.2597503605427412*S*       eta**2*(chi1 - chi2) - 0.257787704938017*eta**2*       (1 + 8.75928187268504*eta**2)*(chi1 - chi2)

    @staticmethod
    def amp_insp2_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp2 for mode (2,1)."""
        return _nonnegative_sqrt(1 - 4*eta)*(0.05511628628738656 - 0.12579599745414977*eta +         2.831411618302815*eta**2 - 14.27268643447161*eta**3 +         28.3307320191161*eta**4) + _nonnegative_sqrt(1 - 4*eta)*       (S**2*(-0.009520793625590234 + 0.11647097598638301*eta -           0.8321002363767337*eta**2) + S*(-0.008692738851491525 +           0.09512553997347648*eta - 0.34097692884809583*eta**2) +         S**3*(-0.006383232900211555 - 0.13099477081654226*eta**2)) -       0.2962753588645467*eta**2*(1 + 1.3993978458830476*eta**2)*       (chi1 - chi2) - 0.17100612756133535*S*eta**2*       (1 + 18.974303741922743*_nonnegative_sqrt(1 - 4*eta)*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_insp3_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp3 for mode (2,1)."""
        return _nonnegative_sqrt(1 - 4*eta)*(0.059110044024271766 - 0.0024538774422098405*eta +         0.2428578654261086*eta**2) + _nonnegative_sqrt(1 - 4*eta)*       (S**2*(-0.006952154764487412 - 0.1270257962053741*eta**2) +         S*(-0.007044339356171244 - 0.016643018304732624*eta**2) +         S**3*(-0.007685497720848459 + 0.004623467175906347*eta**2)) -       0.3172310538516028*eta**2*(1 - 2.9155919835488024*eta**2)*       (chi1 - chi2) - 0.11975485688200693*S*eta**2*       (1 + 17.27626751837825*_nonnegative_sqrt(1 - 4*eta)*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_int1_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int1 for mode (2,1)."""
        return _nonnegative_sqrt(eta - 4*eta**2)*(21.256776327599113 - 25.594352690383843*         eta + 30.14761650482866*eta**2) + _nonnegative_sqrt(eta - 4*eta**2)*       (S**3*(0. - 4.881427482271166*eta) +         S*(-11.26204498563276 - 1.1798437990445072*eta) +         S**2*(-1.8167045597937679 + 6.344825546437451*eta)) -       31.60048733143782*eta**2*(1 + 2.1502870640831855*eta**2)*       (chi1 - chi2) - 3.6366100759176696*(1 - 4*eta)*eta*       (chi1 - chi2)**2

    @staticmethod
    def amp_int2_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int2 for mode (2,1)."""
        return _nonnegative_sqrt(eta - 4*eta**2)*(19.154450657080044 - 21.135962294383088*         eta + 29.74256594428577*eta**2) + _nonnegative_sqrt(eta - 4*eta**2)*       (S*(-12.766814596085732 - 2.9131849820250415*eta) +         S**2*(-2.123816950673979 + 6.0065715496619*eta)) -       25.856046423804255*eta**2*(1 + 5.7871199275552*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_int0_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int0 for mode (2,1)."""
        return 0.872895771366973 + 441.76285124642845*eta -       24617.068739152524*eta**2 + 518054.9485981792*eta**3 +       S**2*(-0.2687302438646897 - 113.2972558250989*eta +         2640.9199254296354*eta**2) + S*(-0.07204945394855851 -         173.67847091983126*eta + 3571.039358823004*eta**2)

    @staticmethod
    def amp_dint0_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit DInt0 for mode (2,1)."""
        return -0.8535048463050732 - 93.1876950411214*eta +       13641.071903017495*eta**2 - 337621.44851304166*eta**3 +       S**2*(-0.26539816223182355 + 189.63365484152848*eta -         8172.057025783848*eta**2) + S*(-1.2067842398131878 +         77.26350785961218*eta - 1972.284151572111*eta**2)

    @staticmethod
    def amp_alambda_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit ALambda for mode (2,1)."""
        return _nonnegative_sqrt(eta - 4*eta**2)*(0.007349833876686359 -         0.0012619735607202085*eta + 0.01042318959002753*eta**2) +       _nonnegative_sqrt(eta - 4*eta**2)*(S*(-0.0048396457425702024 -           0.054621206928483663*eta**2) +         S**3*(0. + 0.020360826886107204*eta**2) +         S**2*(-0.0013927779195756038 + 0.02595660494955219*eta**2)) -       0.018115657394753674*eta**2*(1 - 10.539795474715346*_nonnegative_sqrt(1 - 4*eta)*         eta**2)*(chi1 - chi2)

    @staticmethod
    def amp_lambda_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Lambda for mode (2,1)."""
        return 0.5566284518926176 + 0.12651770333481904*eta +       1.8084545267208734*eta**2 + S**2*(0.2742026346233667 -         0.319247481175182*eta - 3.496995664461796*eta**2) +       S*(0.2907492222665154 + 0.05931724360647139*eta -         2.1011113994370345*eta**2) + 1.0122975748481835*_nonnegative_sqrt(1 - 4*eta)*       eta**2*(chi1 - chi2)

    @staticmethod
    def amp_sigma_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Sigma for mode (2,1)."""
        return 1.2922261617161441 + 0.0019318405961363861*       eta + S*(0.0492798255110865 - 0.6703778360948939*eta +         2.6625014134659777*eta**2) + 1.2001101665670462*_nonnegative_sqrt(1 - 4*eta)*       eta**2*(chi1 - chi2) + 1.2001101665670462*_nonnegative_sqrt(1 - 4*eta)*       eta**2*(chi1 - chi2)**2

    @staticmethod
    def amp_insp1_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp1 for mode (3,2)."""
        return jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.024621376891809633 - 0.09692699636236377*eta +         2.7200998230836158*eta**2 - 16.160563094841066*eta**3 +         32.930430889650836*eta**4) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S**2*(-0.00362963820787208 + 0.15878436411950508*eta -           0.7550771176087877*eta**3) + S**3*(0.016978054142418417 +           0.3381436950547799*eta**3) + S**4*(0.029465887557447824 -           0.14702881770475767*eta + 2.5663386135613093*eta**3) +         S*(0.008522695567479373 + 0.2428055404083171*eta -           1.1104639098529456*eta**2 + 4.649438233164449*eta**3)) +       0.1465612311350642*S*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) +       215.06940561269835*_nonnegative_sqrt(1 - 4*eta)*eta**8*(chi1 - chi2) -       0.007061187955941243*eta**2*(1 - 2.024701925508361*eta**2)*       (chi1 - chi2)**2

    @staticmethod
    def amp_insp2_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp2 for mode (3,2)."""
        return jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.024621376891809633 - 0.09692699636236377*eta +         2.7200998230836158*eta**2 - 16.160563094841066*eta**3 +         32.930430889650836*eta**4) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S**2*(-0.00362963820787208 + 0.15878436411950508*eta -           0.7550771176087877*eta**3) + S**3*(0.016978054142418417 +           0.3381436950547799*eta**3) + S**4*(0.029465887557447824 -           0.14702881770475767*eta + 2.5663386135613093*eta**3) +         S*(0.008522695567479373 + 0.2428055404083171*eta -           1.1104639098529456*eta**2 + 4.649438233164449*eta**3)) +       0.1465612311350642*S*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) +       215.06940561269835*_nonnegative_sqrt(1 - 4*eta)*eta**8*(chi1 - chi2) -       0.007061187955941243*eta**2*(1 - 2.024701925508361*eta**2)*       (chi1 - chi2)**2

    @staticmethod
    def amp_insp3_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp3 for mode (3,2)."""
        return (jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.023159999980228326 - 0.06572921544866189*eta -          0.09869919162707179*eta**2 + 0.07171173224831695*eta**3))/       (1 - 3.7657611696878934*eta) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S**2*(0.00347699923233349 + 0.05592992411391445*eta -           0.008730335213434078*eta**2) + S**3*(0.012079324401547036 +           0.04615964369925028*eta**2) + S*(0.011073381681404716 +           0.5440307361144314*eta**2) + S**4*(0.016323101357296865 -           0.15666140197050316*eta + 0.6703688097531086*eta**2)) +       0.10464021067936541*S*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) -       417.3604094454253*_nonnegative_sqrt(1 - 4*eta)*eta**8*(chi1 - chi2) -       0.020140175824954427*eta**2*(1 - 12.675522774051249*eta**2)*       (chi1 - chi2)**2

    @staticmethod
    def amp_int1_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int1 for mode (3,2)."""
        return jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.019069933430190773 - 0.19396651989685837*eta +         11.95224600241255*eta**2 - 158.90113442757382*eta**3 +         1046.65239329071*eta**4 - 3476.940285294999*eta**5 +         4707.249209858949*eta**6) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S**2*(-0.0038263656140933152 + 2.3753413452420133*eta**2 -           7.925985329178953*eta**3) + S**3*(0.018963579407636953 +           0.4978494265612399*eta**3) + S**4*(0.02738043801805805 -           0.9085620866763245*eta**2 + 5.225566502711913*eta**3) +         S*(0.0046910348789512895 + 0.40231360805609434*eta -           1.955352354930108*eta**2 + 7.977057990568723*eta**3)) +       0.27352193588394114*S*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) +       1635.4240644598524*_nonnegative_sqrt(1 - 4*eta)*eta**8*(chi1 - chi2) +       0.058560321425018165*eta**2*(1 - 19.936477485971217*eta**2)*       (chi1 - chi2)**2

    @staticmethod
    def amp_int2_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int2 for mode (3,2)."""
        return jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*(5.9418458424054235 -         31.905244419036787*eta + 271.1056329988318*eta**2 -         2113.965233486898*eta**3 + 6214.038393898584*eta**4) +       jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*(S*(-2.726472456645038 +           98.08119212251981*eta - 1595.8027347570667*eta**2 +           10581.664858726683*eta**3 - 21488.25117198268*eta**4) +         S**2*(2.9454485454761827 - 119.88112323140916*eta +           1686.7137359336039*eta**2 - 8474.190197512324*eta**3 +           13866.428366595079*eta**4) + S**3*(3.5684571473795095 -           145.5079981415436*eta + 2139.8290160628144*eta**2 -           11680.937129551317*eta**3 + 20863.270079587106*eta**4)) +       0.0038732029045487884*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def amp_int0_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int0 for mode (3,2)."""
        return 0.24794156582503746 + 115.81823862983131*eta -       6626.167995915723*eta**2 + 141004.29332593994*eta**3 +       S*(0.21144389781375486 + 35.10041265469985*eta -         1794.2301585086836*eta**2) + S**2*(0.2781735549493081 -         37.038950686633*eta + 1258.628375238807*eta**2) +       S**3*(0.23428222791962147 - 63.98011009365722*eta +         2118.2135628999335*eta**2)

    @staticmethod
    def amp_dint0_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit DInt0 for mode (3,2)."""
        return -0.3391808620221253 - 14.604141885467747*eta +       3694.1706648870427*eta**2 - 95482.02951271653*eta**3 +       S**2*(-1.2356159239385598 + 188.2052849646004*eta -         6332.389157828061*eta**2) + S*(-1.284450209079395 +         149.04865679660236*eta - 5018.762853306417*eta**2)

    @staticmethod
    def amp_alambda_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit ALambda for mode (3,2)."""
        return 0.0000961373238758885 + 0.051931503056087645*eta -       1.9588543209748939*eta**2 + 45.48261444575759*eta**3 -       624.503508952714*eta**4 + 5023.975867265672*eta**5 -       23254.883056784765*eta**6 + 57103.54590319375*eta**7 -       57373.109075418375*eta**8 + S**2*(-0.000043667015308654945 -         0.008613637625472218*eta + 0.023899142473578886*eta**2 +         0.27415927607062834*eta**3 - 0.9031990932644143*eta**4) +       S*(-0.00003549735163676504 - 0.008295846576936152*eta +         0.06672653429702645*eta**2 - 0.6832451831750879*eta**4) +       S**3*(-0.00004006237792974902 - 0.010228849304319637*eta +         0.10609858524315585*eta**2 - 0.36612035981284025*eta**3 +         0.4506555292261545*eta**4) + 0.18527682139719162*S*eta**2*       (1 - 9.588465932644539*eta + 22.385733051197636*eta**2)*       (chi1 - chi2) + eta**6*(-175.70216708421992 +         1434.281911022591*eta - 2431.323183673857*eta**2 -         1978.252612764647*eta**3)*(chi1 - chi2) -       0.1129183488923664*eta**2*(1 - 9.302928432401565*eta +         21.04582020713889*eta**2)*(chi1 - chi2)**2

    @staticmethod
    def amp_lambda_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Lambda for mode (3,2)."""
        return (jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.7483576461444402 - 6.601172011472086*eta +          15.041074031450826*eta**2))/(1 - 8.763310467475133*eta +         19.460466568409927*eta**2) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S*(0.022780861105257896 - 7.744918955305337*eta +           98.83017893639101*eta**2 - 229.67169290324026*eta**3) +         S**2*(0. + 2.9012202332428583*eta - 7.292998534944377*eta**3) +         S**3*(0.22628217752352817 + 6.4309577443439325*eta**3)) -       1.0164739399698035*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) -       4.2628695434147295*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)**2

    @staticmethod
    def amp_insp1_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp1 for mode (3,3)."""
        return (_nonnegative_sqrt(1 - 4*eta)*(0.14174542816732136 + 0.37703538922015295*eta -          1.2953040762844794*eta**2))/(1 - 2.380527623106869*eta) +       _nonnegative_sqrt(1 - 4*eta)*(S*(0. + 2.128974052579173*eta -           7.898219960437845*eta**2) + S**2*(0.0575999212331518 -           0.3033779841906219*eta**2) + S**3*(0.12732326642391448 -           1.8837167749776262*eta + 5.928292559702788*eta**2)) +       0.4353380004310881*eta**2*(1 - 9.272362349860394*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_insp2_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp2 for mode (3,3)."""
        return _nonnegative_sqrt(1 - 4*eta)*(0.21364255412404595 - 0.7570509486645325*eta +         25.49726191445782*eta**2 - 305.6711306250643*eta**3 +         2006.4739824300186*eta**4 - 6573.827442800209*eta**5 +         8511.227063912822*eta**6) + _nonnegative_sqrt(1 - 4*eta)*       (S*(0.020500959266784635 + 1.607996092610713*eta -           6.215311401878405*eta**2) + S**2*(0.05768802659073407 -           0.4407201081412575*eta + 1.6518802058417443*eta**2) +         S**3*(0.10190109823796438 - 2.0189323002941464*eta +           8.099228434549993*eta**2)) + 0.4046803441986717*eta**2*       (1 - 11.424655821801338*eta**2)*(chi1 - chi2)

    @staticmethod
    def amp_insp3_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp3 for mode (3,3)."""
        return (S**3*(0.05315163310055228 - 0.26294356475755726*eta) +         S**2*(0.048394597995531904 - 0.1992792089282844*eta) +         S*(0.045278587020896065 + 0.15796632912496153*eta))*       _nonnegative_sqrt(1 - 4*eta) + _nonnegative_sqrt(1 - 4*eta)*(0.23630540622294194 +         0.28839457061276524*eta - 10.29181059914425*eta**2 +         164.14987132139012*eta**3 - 1159.1790202271895*eta**4 +         3941.157229416356*eta**5 - 5157.600550999997*eta**6) +       0.23519060494262717*eta**2*(1 - 11.86310290734459*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_int1_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int1 for mode (3,3)."""
        return _nonnegative_sqrt(eta - 4*eta**2)*(27.92765242485774 - 133.56611389260303*         eta + 974.8550901501314*eta**2 - 3744.785831952631*eta**3 +         5621.897260910284*eta**4) + _nonnegative_sqrt(eta - 4*eta**2)*       (S**2*(5.089079109477854 - 37.91880996555659*eta +           86.54385209398798*eta**2) + S*(7.34059425787106 -           60.145815223458676*eta + 159.3796614719456*eta**2)) +       13.45782954129836*S*eta**4*(chi1 - chi2) +       1420.23885817822*eta**6*(chi1 - chi2)

    @staticmethod
    def amp_int2_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int2 for mode (3,3)."""
        return _nonnegative_sqrt(eta - 4*eta**2)*(20.162169689041896 - 18.666422946967767*         eta + 53.04107631052986*eta**2) + _nonnegative_sqrt(eta - 4*eta**2)*       (S*(4.716262418092889 - 76.55185215072702*eta +           515.3316005298788*eta**2 - 1111.9346236019971*eta**3) +         S**2*(5.080208137626593 - 86.42142446536636*eta +           486.9724744307628*eta**2 - 808.262506994693*eta**3)) +       1635.4394234451456*(1 - 0.16768464526251092*_nonnegative_sqrt(1 - 4*eta))*eta**6*       (chi1 - chi2)

    @staticmethod
    def amp_int0_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int0 for mode (3,3)."""
        return 1.5852399637975103 + 549.5183711492834*eta -       34257.76380246282*eta**2 + 743142.8286902909*eta**3 +       S*(0.7436306553052215 - 89.49451655594794*eta +         1185.165432591372*eta**2) + S**2*(0.42530249797257236 -         174.57306465486613*eta + 6510.983041407188*eta**2)

    @staticmethod
    def amp_dint0_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit DInt0 for mode (3,3)."""
        return -4.691600252198376 + 101.4338937535679*eta +       9262.994550540048*eta**2 - 310993.1309846956*eta**3 +       S*(-4.1982323942191115 + 734.7118618746312*eta -         28714.90419206062*eta**2) + S**2*(-0.4098659551231475 +         292.0456626070157*eta - 5100.093360692769*eta**2)

    @staticmethod
    def amp_alambda_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit ALambda for mode (3,3)."""
        return (S**2*(0.003561744165171048 - 0.01818631702607634*eta) +         S*(0.007585898058607901 - 0.013132320758494444*eta))*       _nonnegative_sqrt(eta - 4*eta**2) + _nonnegative_sqrt(eta - 4*eta**2)*       (0.013700854227665185 + 0.012027324273217743*eta +         0.08980955088895569*eta**2) + 0.05745949361626236*S*eta**4*       (1 - 206.70909462226626*_nonnegative_sqrt(1 - 4*eta)*eta**2)*       (chi1 - chi2) + 0.09802218411554885*eta**4*       (1 - 43.54102580612641*_nonnegative_sqrt(1 - 4*eta)*eta**2)*       (chi1 - chi2)

    @staticmethod
    def amp_lambda_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Lambda for mode (3,3)."""
        return 0.7435306475478924 + S*(0.19457194111990658 -         0.4809350398289311*eta) + S**3*(0. - 0.1988991467974821*eta) +       S**2*(0.07564220573555203 + 0.17261430318577403*eta) -       0.06688558533374556*eta + 1.471989765837694*eta**2 +       1.8881959341735146*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def amp_insp1_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp1 for mode (4,4)."""
        return jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*(0.06190013067931406 + 0.1928897813606222*eta +         1.9024723168424225*eta**2 - 15.988716302668415*eta**3 +         35.21461767354364*eta**4) + jnp.sqrt(jnp.maximum(1 - 3*eta, 0.0))*       (S*(0.011454874900772544 + 0.6600413908621988*eta -           2.1705970511116486*eta**2) + S**2*(0.044702230915643903 +           0.12149520289658673*eta - 0.6512813450832168*eta**2) +         S**3*(0.07327810908370004 - 0.4482406547006759*eta +           1.1237234702682313*eta**2)) + 0.4766851579723911*eta**2*       (1 - 15.950025762198988*eta**2)*(chi1 - chi2) +       0.127900699645338*eta**2*(1 - 15.79329306044842*eta**2)*       (chi1 - chi2)**2

    @staticmethod
    def amp_insp2_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp2 for mode (4,4)."""
        return 0.08406011695496626 - 0.1469952725049322*eta +       0.2997223283799925*eta**2 - 1.2910560244510723*eta**3 +       S*(0.023924074703897662 + 0.26110236039648027*eta -         1.1536009170220438*eta**2) + S**2*(0.044797272997526706 -         0.14398688588718014*eta + 0.05736387085230215*eta**2) +       S**3*(0.06028104440131858 - 0.4759412992529712*eta +         1.1090751649419717*eta**2) + 0.2648241309154185*_nonnegative_sqrt(1 - 4*eta)*       eta**2*(chi1 - chi2) + 0.10346324686812074*eta**2*       (1 - 16.135903382018213*eta**2)*(chi1 - chi2)**2

    @staticmethod
    def amp_insp3_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Insp3 for mode (4,4)."""
        return 0.08212436946985402 + S**2*(0.03949133192324476 -         0.12754920943544915*eta) + S**3*(0.034468577523793176 -         0.1119901209970158*eta) + S*(0.03488890057062679 -         0.08968833480313296*eta) - 0.025332770704783136*eta -       3.2466088293309885*eta**2 + 28.404235115663706*eta**3 -       111.36325359782991*eta**4 + 157.05954559045156*eta**5 +       0.2062291124580944*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def amp_int1_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int1 for mode (4,4)."""
        return jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*(10.804555518381164 - 72.3834734399584*eta +         540.0541240482851*eta**2 - 2612.999845214264*eta**3 +         4779.096001663425*eta**4) + jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*       (S**2*(3.097395982217429 - 39.31284390368826*eta +           106.9129562723712*eta**2) + S*(4.2633625314212145 -           47.94914754514521*eta + 119.70401520575764*eta**2)) +       3.001401833124412*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2) +       0.7262636326998003*(1 - 4*eta)*eta*(chi1 - chi2)**2

    @staticmethod
    def amp_int2_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int2 for mode (4,4)."""
        return jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*(9.020721305469886 - 53.22188349231127*eta +         508.07176447172276*eta**2 - 3194.0620894511526*eta**3 +         6769.927439234588*eta**4) + jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*       (S**2*(2.3744342198527484 - 25.286684856422735*eta +           64.74544118094356*eta**2) + S*(3.2565916700919675 -           38.389225546513565*eta + 96.41777041220985*eta**2)) +       3.233759337559542*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def amp_int0_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Int0 for mode (4,4)."""
        return 0.5664660641971224 + 185.58965113823874*eta -       11458.768824989507*eta**2 + 249386.7511724409*eta**3 +       S**3*(0.4423803798742513 - 23.60535149579996*eta -         994.9241585715828*eta**2) + S*(0.17417687762107817 -         9.365114803167128*eta + 703.2622732011035*eta**2) +       S**2*(0.201692297830482 - 62.13147149352514*eta +         2833.5738711424974*eta**2)

    @staticmethod
    def amp_dint0_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit DInt0 for mode (4,4)."""
        return -1.796444922382065 + 111.51170611049032*eta -       1728.7493675776548*eta**2 + S*(-1.8421198606139237 +         312.34531117524085*eta - 11235.484645624338*eta**2) +       S**2*(-0.36655273031432567 + 128.64488103364164*eta -         2927.019210835522*eta**2)

    @staticmethod
    def amp_alambda_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit ALambda for mode (4,4)."""
        return jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*(0.007904587819112175 +         0.09558474985614368*eta - 2.6638033973597746*eta**2 +         28.298192768381554*eta**3 - 136.10446022757955*eta**4 +         233.23167528016833*eta**5) + jnp.sqrt(jnp.maximum(eta - 3*eta**2, 0.0))*       (S**2*(0.004122811292229324 - 0.06166686913913691*eta +           0.4139026619690879*eta**2 - 0.9656098473922222*eta**3) +         S**3*(0.000657528128497184 + 0.014107365722576927*eta -           0.1389170612199015*eta**2 + 0.19708289555425246*eta**3) +         S*(0.0049703757209330025 - 0.2945455034809188*eta**2 +           0.9225758392294605*eta**3)) + 0.00659873279539475*       _nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def amp_lambda_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Amplitude fit Lambda for mode (4,4)."""
        return (1 - 3*eta)*(0.7784823183691343 + 34.46044836457201*eta -         1514.0269701450463*eta**2 + 26746.055755132693*eta**3 -         217137.36747452553*eta**4 + 821240.0917610931*eta**5 -         1.1528621947893575e6*eta**6) + (1 - 3*eta)*       (0.04916802087602157*(-1.9958484192113994 + 13.539997038369126*eta -           3.6208750340313913*eta**2) +         ((-0.049168020876021526 + 0.028838699493050773*S)*          (-1.8291582953092906 + 113.81460909267167*eta**2))/         (1 - 0.881817210461074*S)) + 0.8535730715294357*_nonnegative_sqrt(1 - 4*eta)*       eta**2*(chi1 - chi2) + 0.959943270591552*eta**2*       (chi1 - chi2)**2

    # -------------------------------------------------------------------------
    # Phase Derivative and Shift Fits
    # -------------------------------------------------------------------------
    @staticmethod
    def phase_p1_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p1 for mode (2,1)."""
        return 6.713584981247434 + 536.7465954562813*eta -      2151.648363721454*eta**2 + 4128.014499582183*eta**3 +      S**2*(10.45011045204613 + 485.738458747382*eta -        2463.1612459443427*eta**2) +      S*(3.7361694179241787 - 513.1752235722759*eta +        994.0961950836344*eta**2) + S**3*(7.918023212219605 -        391.11084366980054*eta + 1855.0623015585954*eta**2)

    @staticmethod
    def phase_p2_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p2 for mode (2,1)."""
        return 0.9186799123631688 + 267.41497120115645*eta -      695.2965191341449*eta**2 + 1095.721690358062*eta**3 +      S**3*(5.779095253218805 - 85.81111830894119*eta**2) +      S*(0.7083999858841953 - 57.85416146458766*eta -        56.13860804167412*eta**2) + S**2*(6.1047188723865435 -        98.1439571306194*eta + 302.72479382413604*eta**2)

    @staticmethod
    def phase_p3_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p3 for mode (2,1)."""
        return 3241.679716506053 + 890.0161988778857*eta -      28651.884268147278*eta**2 + 369152.5761872605*eta**3 -      2.288322744672021e6*eta**4 + 6.825330367257056e6*eta**5 -      7.862540519829421e6*eta**6 +      S*(-2.248403360663146 + 154.9525194413342*eta -        445.49612870586583*eta**2) +      S**2*(3.226033542348295 + 49.00082122972584*eta -        97.07004678042274*eta**2) +      S**4*(92.6653678983193 - 880.8042135631449*eta +        1638.8248034025835*eta**2) + S**3*(117.91330329331689 -        987.0640570773274*eta + 1819.2617614289438*eta**2)

    @staticmethod
    def phase_p4_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p4 for mode (2,1)."""
        return 80.80277127955382 - 84.33906423380039*eta +      280.58914116514813*eta**2 - 627.4668183326488*eta**3 +      S*(-28.58390842984726 + 312.0582615033748*eta -        671.9216451681352*eta**2) + S**2*(-42.76965773556711 +        243.76237159870723*eta + 113.29478855038954*eta**2) +      S**4*(154.4099200825563 - 1423.0506016731827*eta +        2549.984519911076*eta**2) + S**3*(137.93938381901975 -        1448.1508794904569*eta + 3400.4248774442767*eta**2)

    @staticmethod
    def phase_p5_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p5 for mode (2,1)."""
        return 3102.3564223499015 + 315.91066557629966*eta -      1688.2590760607561*eta**2 + 3635.760014085031*eta**3 +      S**2*(-49.54347011573923 + 740.4879908500247*eta -        2618.97920206585*eta**2) + S*(-23.095855510776154 +        396.86291758066216*eta - 1434.737847807198*eta**2) +      S**3*(40.70544724730244 - 524.4144885267008*eta +        1939.8269034245006*eta**2) +      S**4*(81.83790112385142 - 1158.0381692620924*eta +        4234.694213429878*eta**2)

    @staticmethod
    def phase_p6_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p6 for mode (2,1)."""
        return 3089.1812457904257 + 4.891940750086265*eta +      190.00802952275086*eta**2 - 255.24462180679632*eta**3 +      S**2*(-18.892851581451023 + 757.7129575060637*eta -        3483.046073122091*eta**2) +      S*(2.9699675868171687 + 11.108562731470132*eta -        186.6082638757012*eta**2) + S**3*(-24.61929877822828 +        549.0848769138178*eta**2) + S**4*(-12.879829033533502 -        1023.1645979988623*eta + 5570.044795875226*eta**2)

    @staticmethod
    def phase_lambda_21(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Inspiral phase correction Lambda_fit for mode (2,1)."""
        return 12.645407036054712 - 35.63119580769972*eta +      564.4350269712382*eta**2 - 1274.1419388751985*eta**3 +      S**2*(-23.307589716645126 + 469.49465957041957*eta -        1487.126225327994*eta**2) +      S*(5.383023150979401 + 32.96520526294001*eta -        216.93245668902682*eta**2) + S**3*(27.283832107900903 -        438.89318378637194*eta + 1145.5488137885761*eta**2)

    @staticmethod
    def phase_p1_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p1 for mode (3,2)."""
        return 3.2589335180457386 + 433.411565457014*eta -      4485.432715110435*eta**2 + 29016.448592832847*eta**3 -      88552.62750206042*eta**4 + 103186.40538667778*eta**5 +      S*(4.471413225013436 - 436.9878258970284*eta +        2643.064065516432*eta**2 - 6896.057216022059*eta**3) +      S**2*(7.628014707255881 + 92.73370230026934*eta**2 +        130.27375207129455*eta**3)

    @staticmethod
    def phase_p2_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p2 for mode (3,2)."""
        return 0.9567025409806279 + 673.2063582087092*eta -      5725.473497547885*eta**2 + 35198.507957865804*eta**3 -      103125.54870074887*eta**4 + 111478.93846090854*eta**5 +      S**3*(10.433273342717518 - 298.01588879837806*eta**2) +      S**2*(15.328654933480447 - 107.66384514522042*eta +        208.74600587946276*eta**2) +      S*(1.9269201997207677 - 207.9811872736127*eta +        298.54791312133625*eta**2)

    @staticmethod
    def phase_p3_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p3 for mode (3,2)."""
        return 109.07716152298035 + 2785.542016518753*eta -      78062.57052206424*eta**2 + 891750.0527530116*eta**3 -      5.145287326482038e6*eta**4 + 1.4775918913462486e7*eta**5 -      1.6801372403691534e7*eta**6 +      S**2*(477.5569063139351 - 8677.280336745405*eta +        50545.83285154399*eta**2 - 95157.22415201803*eta**3) +      S**3*(515.8242307486456 - 9180.400653230974*eta +        49492.36593745136*eta**2 - 84077.52226141123*eta**3) +      S*(29.655038754712862 - 3852.9692012198666*eta**2 +        12665.471965100938*eta**3)

    @staticmethod
    def phase_p4_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p4 for mode (3,2)."""
        return 3307.4901981197413 - 476.9090851445152*eta -      5980.371942864755*eta**2 + 127609.75402360641*eta**3 -      919107.7003332148*eta**4 + 2.8639341606409485e6*eta**5 -      3.2636172954112873e6*eta**6 + S**2*(-43.88226172471013 +        564.3583658777421*eta - 1699.8758810715133*eta**2) +      S**3*(43.88124130414212 - 318.2624830812621*eta +        85.8911817645298*eta**2) + S*(-5.025525637537955 -        278.2775052295507*eta + 1275.2511331683327*eta**2) +      S**4*(106.04745747402781 - 1426.1986941886587*eta +        4239.295307519904*eta**2)

    @staticmethod
    def phase_p5_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p5 for mode (3,2)."""
        return 214.8861570945166 + 37.618586205935856*eta -      15090.74877118785*eta**2 + 146859.8048198116*eta**3 -      519439.0973630157*eta**4 + 608737.1033556702*eta**5 +      S**3*(-680.6074097397208 + 6341.509843910642*eta -        14572.037054770519*eta**2) +      S**4*(-91.4153174957906 + 894.4559718012385*eta -        1975.603770576868*eta**2) + S**2*(48.171849236398344 -        1184.2528420322194*eta**2) +      S*(635.1670556901358 - 6379.029255504343*eta +        15260.831394229439*eta**2)

    @staticmethod
    def phase_p6_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p6 for mode (3,2)."""
        return 48.460105733891275 + 3490.6705492125598*eta -      117183.75947754481*eta**2 + 1.9464435908970707e6*eta**3 -      1.8300214618414816e7*eta**4 + 9.85627669791169e7*eta**5 -      2.783199904079779e8*eta**6 + 3.1586641575325894e8*eta**7 +      S*(-24.776436154670087 + 572.5076095394872*eta -        1519.9579815209986*eta**2) + S**2*(186.18755396615907 -        1582.9144970634259*eta + 3575.660058382156*eta**2) +      S**3*(245.51469777691742 - 2281.4564921725205*eta +        5438.947115977561*eta**2)

    @staticmethod
    def phase_lambda_32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Inspiral phase correction Lambda_fit for mode (3,2)."""
        return 9.923848991260856 + 31.571277471465866*eta -      663.8785669033999*eta**2 + 3238.8788430103527*eta**3 -      8227.832469492027*eta**4 + S**3*(0.40218061798997906 +        54.4132981108995*eta - 130.17465165145506*eta**2) +      S**2*(-0.37376441561353607 - 133.2917016916967*eta +        330.4288288898059*eta**2) +      S*(-5.122708178962234 - 19.406706514605503*eta +        350.8490121993012*eta**2)

    @staticmethod
    def phase_p1_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p1 for mode (3,3)."""
        return 3.4845963792706716 + 563.1328192071531*eta -      6330.310694310805*eta**2 + 43999.89007734047*eta**3 -      147011.49484776487*eta**4 + 189156.48682633752*eta**5 +      S*(4.882213653619051 - 264.69022097541*eta + 146.11633414603656*        eta**2) + S**2*(9.997837297421023 - 48.37016933171186*eta +        308.9821522046684*eta**2)

    @staticmethod
    def phase_p2_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p2 for mode (3,3)."""
        return 0.7866836424968715 + 475.2260703744564*eta -      4193.115068966735*eta**2 + 26892.103234554656*eta**3 -      86713.26078917665*eta**4 + 108239.27939052392*eta**5 +      S**2*(1.8769165701644421 + 4.423647377221832*eta -        63.73726929993851*eta**2) +      S*(9.442208125070376 - 198.9998545433253*eta +        207.40640888447663*eta**2)

    @staticmethod
    def phase_p3_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p3 for mode (3,3)."""
        return 3321.829905997003 + 1796.0287656811952*eta -      52406.11599270297*eta**2 + 605028.02816514*eta**3 -      3.52531806148964e6*eta**4 + 1.0179857736280283e7*eta**5 -      1.1565927741497997e7*eta**6 +      S*(373.6670939145829 - 5817.728339076129*eta +        29741.185576602686*eta**2 - 51519.60550389719*eta**3) +      S**2*(368.5404496413542 - 5389.1225913206845*eta +        26535.30440500079*eta**2 - 44481.97048317842*eta**3)

    @staticmethod
    def phase_p4_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p4 for mode (3,3)."""
        return 82.38829492576497 + 2457.178768399354*eta -      57545.90636397398*eta**2 + 601571.8328396642*eta**3 -      3.27684112065871e6*eta**4 + 9.007305665473843e6*eta**5 -      9.862302028151235e6*eta**6 + S**3*(310.4890231034396 -        6097.198267646682*eta + 37035.13005405912*eta**2 -        71685.04267511083*eta**3) + S**2*(279.07317612941614 -        4803.0811901171965*eta + 26986.493234675127*eta**2 -        49980.966076247416*eta**3) +      S*(-2.257599389384408 + 574.2018960445909*eta -        5923.089020709148*eta**2 + 14146.559540725193*eta**3)

    @staticmethod
    def phase_p5_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p5 for mode (3,3)."""
        return 125.14406491249947 - 2804.2102789014507*eta +      54568.05992501175*eta**2 - 560540.359212911*eta**3 +      3.111436316251264e6*eta**4 - 8.824893943746094e6*eta**5 +      1.000862925963675e7*eta**6 + S**3*(23.590699826951294 -        112.96544787476664*eta + 70.27042475051852*eta**2) +      S*(-2.1271537973118617 + 0.6504492078510683*eta +        80.40329509486911*eta**2) +      S**2*(30.6379267023235 - 264.3990411024577*eta +        668.4615039511533*eta**2)

    @staticmethod
    def phase_p6_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p6 for mode (3,3)."""
        return 2.8336291511250957 + 1758.9390417308175*eta -      36424.667772722496*eta**2 + 384459.14257636265*eta**3 -      2.1667865686701844e6*eta**4 + 6.198587737488048e6*eta**5 -      7.063678471118015e6*eta**6 + S**2*(44.97467481865522 -        776.3508431624222*eta + 4734.942039053759*eta**2 -        9480.110786789375*eta**3) +      S*(1.4704772234862196 - 16.40504251492649*eta +        501.3996485757175*eta**2 - 1849.5827678963567*eta**3) +      S**3*(-11.198942714564465 + 586.5587276252753*eta -        4994.816604177315*eta**2 + 11844.541063193587*eta**3)

    @staticmethod
    def phase_lambda_33(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Inspiral phase correction Lambda_fit for mode (3,3)."""
        return 4.1138398568400705 + 9.772510519809892*eta -      103.92956504520747*eta**2 + 242.3428625556764*eta**3 +      (S*(-0.13253553909611435 + 24.931781913461002*eta -         97.38692036588772*eta**2))/(1 + 0.11322426762297967*S)

    @staticmethod
    def phase_p1_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p1 for mode (4,4)."""
        return 3.6798082752761303 + 545.4667539125608*eta - 5780.5591016515*eta**2 +      39009.9184484156*eta**3 - 129450.82946466094*eta**4 +      169487.01505871397*eta**5 + S**2*(2.209948064434596 +        597.7469998638055*eta**2 - 1687.144608152445*eta**3) +      S*(-19.909021945654636 + 172.999679371611*eta -        2359.2955974900287*eta**2 + 4558.381023974165*eta**3)

    @staticmethod
    def phase_p2_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p2 for mode (4,4)."""
        return 0.6614399165755429 + 495.2212215293292*eta -      4775.35065619013*eta**2 + 32864.33102152535*eta**3 -      111320.57854234384*eta**4 + 142924.6137334795*eta**5 +      S**2*(19.73844619447745 - 353.48605304339594*eta +        2156.169300714138*eta**2 - 4392.692850063938*eta**3) +      S*(7.4428548089089 - 161.46845856306354*eta +        385.0423370370782*eta**3)

    @staticmethod
    def phase_p3_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p3 for mode (4,4)."""
        return 3308.971215686682 + 2353.5759087255055*eta -      66340.14978172812*eta**2 + 777271.6533917319*eta**3 -      4.6443792037099395e6*eta**4 + 1.3779210783169996e7*eta**5 -      1.6080200828482958e7*eta**6 +      S*(198.56859156249408 - 2243.111911061086*eta +        6853.47419911463*eta**2 - 4843.905493232484*eta**3) +      S**2*(31.77100749739782 + 159.73585803173194*eta -        2996.232369476589*eta**2 + 6445.5447299146645*eta**3) +      S**3*(-250.40378007023253 + 3500.514877640155*eta -        15148.216857660074*eta**2 + 19796.770616358222*eta**3)

    @staticmethod
    def phase_p4_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p4 for mode (4,4)."""
        return 63.3429582054951 + 3282.1360086610975*eta - 74804.0385227758*eta**2 +      794694.2502097604*eta**3 - 4.479210426730432e6*eta**4 +      1.2870931871478142e7*eta**5 - 1.4768822156340549e7*eta**6 +      S**3*(508.34734529441977 - 10961.174868055108*eta +        69230.6410681938*eta**2 - 134680.8460726678*eta**3) +      S**2*(379.3101785295918 - 6438.795905980785*eta +        35134.760056160674*eta**2 - 62563.292855487656*eta**3) +      S*(-54.075695079693844 + 2516.3437451451214*eta -        20938.419196414918*eta**2 + 46317.99207533252*eta**3)

    @staticmethod
    def phase_p5_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p5 for mode (4,4)."""
        return 3108.381941299741 + 3722.4589482748784*eta -      119587.7345880769*eta**2 + 1.9214830552993643e6*eta**3 -      1.697956772209648e7*eta**4 + 8.391941719726168e7*eta**5 -      2.1714278020480186e8*eta**6 + 2.282966494194419e8*eta**7 +      S*(217.1926728556804*eta - 925.3482169309893*eta**2) +      S**2*(-17.1674570270549 + 162.49075150758696*eta -        485.04247924432946*eta**2) + S**3*(-23.677914593028696 -        11.77400581367266*eta + 325.80850615897987*eta**2)

    @staticmethod
    def phase_p6_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Intermediate phase derivative at collocation point p6 for mode (4,4)."""
        return 149.59750387408147 - 1915.3121403895739*eta +      28835.00118328937*eta**2 - 237754.95415044806*eta**3 +      1.1500607280723387e6*eta**4 - 3.3836494103068775e6*eta**5 +      4.70040671481685e6*eta**6 + S**3*(-209.05506126248986 +        2112.30594079106*eta - 5035.729102922226*eta**2) +      S**2*(25.3495473063862 - 43.657873964352916*eta -        201.1705602229839*eta**2) +      S*(198.76637515667227 - 1961.7091554982887*eta +        4739.858611035499*eta**2)

    @staticmethod
    def phase_lambda_44(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Inspiral phase correction Lambda_fit for mode (4,4)."""
        return (5.1934918468328055 - 24.955542694509784*eta)/      (1 - 2.74771030126607*eta) + S*(-0.15677624592293554 +        43.75241570601949*eta - 183.0408615217376*eta**2) +      S**2*(0.26739228895125094 - 40.47330401340779*eta +        167.70366422977224*eta**2)

    # -------------------------------------------------------------------------
    # Mode (3,2) Spheroidal Ringdown Phase Fits
    # -------------------------------------------------------------------------
    @staticmethod
    def phase_32_s_dphi_p1(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Phase derivative in spheroidal basis at point 1 (f_ring_22)."""
        return 3169.372056189274 + 426.8372805022653*eta - 12569.748101922158*eta**2 +    149846.7281073725*eta**3 - 817182.2896823225*eta**4 +    1.5674053633767858e6*eta**5 +    S**5*(-61.01931672442576 - 14949.978871519907*eta +      141578.00190209807*eta**2 - 324588.1758876472*eta**3) +    S**2*(-42.88446003698396 - 1946.8401658264224*eta +      24075.31985882056*eta**2 - 61728.62378428906*eta**3) +    S**4*(-24.72321125342808 - 1246.4743873422667*eta +      8851.873422260202*eta**2 - 17627.229389471784*eta**3) +    S*(19.23408352151287 - 1403.9117194979838*eta +      4164.09550826528*eta**2 + 5239.455637618375*eta**3) +    S**3*(13.918237996338371 + 8258.408022796391*eta -      69558.07335097509*eta**2 + 142974.2777470088*eta**3) +    551.0586141675731*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def phase_32_s_dphi_fring32(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Phase derivative in spheroidal basis at f_ring_32."""
        return 3064.689378902289 + 67.27820968023886*eta - 340.65499218757446*eta**2 +    S**2*(86.32937405207333 - 2119.179085503391*eta +      13660.804173755761*eta**2 - 25583.314298758105*eta**3) +    S*(43.58652832653278 - 704.4454291401064*eta +      1403.3768617890412*eta**2 + 2472.6476419567534*eta**3) +    S**3*(-143.9230418566328 + 1922.2284594500547*eta -      9811.222294727015*eta**2 + 16259.911572457446*eta**3) +    S**4*(-215.7449874927435 + 4676.057579480619*eta -      33159.79299552468*eta**2 + 69666.01833764123*eta**3) +    402.63197312684605*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def phase_32_s_dphi_p4(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Phase derivative in spheroidal basis at point 4 (f_ring_32 + 0.5 f_damp_32)."""
        return 3077.0657367004565 + 64.99844502520415*eta -    357.38692756785395*eta**2 +    S**2*(57.38106384558743 - 1644.6690499868596*eta +      11008.881935880598*eta**2 - 19906.416384606226*eta**3) +    S*(34.793450080444714 - 986.7751755509875*eta +      5700.682624203565*eta**2 - 9490.641676924794*eta**3) +    S**3*(-126.02362949830213 + 3169.3397351803583*eta -      26766.730897942085*eta**2 + 62863.79877094988*eta**3) +    S**4*(-169.30909412804587 + 4900.706039920717*eta -      41414.05689348732*eta**2 + 95314.99988114933*eta**3) +    390.5443469721231*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    @staticmethod
    def phase_32_s_deltaphi(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Phase shift delta_Phi_S between 32 and 22 at f_ring_22."""
        return -1.2985547810045444 - 428.1198658825042*eta**2 +    13442.64862884998*eta**3 - 142888.7278161952*eta**4 +    636707.5645170014*eta**5 - 996748.791851156*eta**6 +    S*(0.4936924192713203 - 9.914116434251216*eta +      169.5013542820467*eta**2 - 530.1717573803628*eta**3) +    S**2*(-0.5078684457794266 + 20.41643651531135*eta -      34.23514979896381*eta**2 - 164.3835614153519*eta**3) +    S**4*(1.5276575498978686 + 48.05716169088652*eta -      624.2021132936136*eta**2 + 1610.0532058576669*eta**3) +    S**3*(0.5897873058761964 + 72.78541774024623*eta -      748.3162793405884*eta**2 + 1763.7397712348*eta**3) +    60.03625318565433*_nonnegative_sqrt(1 - 4*eta)*eta**2.5*(chi1 - chi2)

    @staticmethod
    def phase_32_s_deltadphi(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Time shift delta_T_S between 32 and 22 at f_ring_22 + f_damp_22."""
        return 11.851438981981772 + 167.95086712701223*eta -    4565.033758777737*eta**2 + 61559.132976189896*eta**3 -    364129.24735853914*eta**4 + 739270.8814129328*eta**5 +    S**4*(15.585154698977842 - 2292.778112523392*eta +      24793.809334683185*eta**2 - 65993.84497923202*eta**3) +    S**3*(3.4881615575084797 - 2244.4613237912527*eta +      24145.932943269272*eta**2 - 60929.87465551446*eta**3) +    S**2*(-5.949655484033632 - 307.67253970367034*eta +      1334.1062451631644*eta**2 + 3575.347142399199*eta**3) +    S*(9.506768471271634 + 434.31707030999445*eta -      8046.364492927503*eta**2 + 26929.677144312944*eta**3) +    465.7904934097202*_nonnegative_sqrt(1 - 4*eta)*eta**2*(chi1 - chi2)

    # -------------------------------------------------------------------------
    # Ringdown (2,2) Rescaling Phase Fits (Eq. 6.5)
    # -------------------------------------------------------------------------
    @staticmethod
    def phase_rd_alpha_lambda(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Ringdown alpha_lambda_22 parameter fit (Eq. 6.5)."""
        return eta * (-1.1926122248825484 + 2.5400257699690143 * eta - 16.504334734464244 * eta**2 + 27.623649807617376 * eta**3 + S * (0.10345265546549828 - 4.7282481007397825 * eta + 35.803988443700824 * eta**2 - 72.50310678862684 * eta**3) + S**2 * (-0.21477847929548569 + 0.8743576195364631 * eta + 9.700178927988006 * eta**2 - 39.83460092417137 * eta**3) + S**3 * (-0.06417449517826644 + 8.170616575493503 * eta - 77.2346297158916 * eta**2 + 180.8345521274853 * eta**3) - 0.7428134042821221 * _nonnegative_sqrt(1.0 - 4.0 * eta) * eta**2.5 * (chi1 - chi2) + 17.588573345324154 * _nonnegative_sqrt(1.0 - 4.0 * eta) * eta**3.5 * (chi1 - chi2)**2)

    @staticmethod
    def phase_rd_alpha2(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """Ringdown alpha_2_22 parameter fit (Eq. 6.5)."""
        return (0.06508727556277369 + 81.82151902053695 * eta - 496.1922346106388 * eta**2 + 2631.5581174891404 * eta**3 - 8864.303027536791 * eta**4 + 11817.83797276061 * eta**5 + (S * (0.977436415054621 - 4.9556042646000975 * eta + 26.352592393899965 * eta**2)) / (1.0 - 0.7251106626013689 * S) - 28.473834755218146 * _nonnegative_sqrt(1.0 - 4.0 * eta) * eta**2 * (chi1 - chi2) - 27.778150812657373 * _nonnegative_sqrt(1.0 - 4.0 * eta) * eta**3 * (chi1 - chi2)**2)
