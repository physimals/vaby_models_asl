"""
VABY_MODELS_ASL - Resting-state forward model for ASL data
"""
import tensorflow as tf
import numpy as np

from vaby.model import Model, ModelOption
from vaby.utils import ValueList, NP_DTYPE
from vaby.parameter import get_parameter

from . import __version__

class AslRestModel(Model):
    """
    ASL resting state model

    FIXME integrate with oxasl AslImage class?
    """

    OPTIONS = [
        # ASL parameters 
        ModelOption("tau", "Bolus duration", units="s", clargs=("--tau", "--bolus"), type=float, default=1.8),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=False),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption("plds", "Post-labelling delays (for CASL instead of TIs)", units="s", type=ValueList(float)),
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=[1]),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float, default=0),

        # GM tissue properties 
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
        ModelOption("att", "Bolus arrival time", units="s", clargs=("--bat",), type=float, default=1.3),
        ModelOption("attsd", "Bolus arrival time prior std.dev.", units="s", clargs=("--batsd",), type=float, default=None),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
        ModelOption("pc", "Blood/tissue partition coefficient. If only inferring on one tissue, default is 0.9; if inferring on both GM/WM default is 0.98/0.8 respectively. See --pcwm", type=float, default=None),

        # WM tissue properties 
        ModelOption("incwm", "Include WM component at each node", default=False),
        ModelOption("inferwm", "Infer WM parameters at each node", default=False),
        ModelOption("pvcorr", "Partial volume correction - equivalent to incwm and inferwm", default=False),
        ModelOption("t1wm", "WM T1 value", units="s", type=float, default=1.1),
        ModelOption("fwm", "WM perfusion value to use if incwm=True and inferwm=False", type=float, default=0),
        ModelOption("attwm", "WM arterial transit time. Used as prior and initial posterior if inferwm=True, used as fixed value if inferwm=False", clargs=("--batwm",), type=float, default=1.6),
        ModelOption("fcalibwm", "WM perfusion value to use in estimation of effective T1", type=float, default=0.003),
        ModelOption("pcwm", "WM parition coefficient. See --pc", type=float, default=0.8),

        # Blood / arterial properties 
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=1.65),
        ModelOption("artt", "Arterial bolus arrival time", units="s", clargs=("--batart",), type=float, default=None),
        ModelOption("arttsd", "Arterial bolus arrival time prior std.dev.", units="s", clargs=("--batartsd",), type=float, default=None),

        # Inference options 
        ModelOption("artonly", "Only infer arterial component not tissue", type=bool),
        ModelOption("inferart", "Infer arterial component", type=bool),
        ModelOption("infert1", "Infer T1 value", type=bool),
        ModelOption("att_init", "Initialization method for ATT (max=max signal - bolus duration)", default=""),

        # PVE options 
        ModelOption("pvgm", "GM partial volume", type=float, default=1.0),
        ModelOption("pvwm", "WM partial volume", type=float, default=0.0),
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        if self.plds is not None:
            self.tis = [self.tau + pld for pld in self.plds]

        if self.tis is None and self.plds is None:
            raise ValueError("Either TIs or PLDs must be given")

        # Only infer ATT with multi-time data 
        self.inferatt = (len(self.tis) > 1)

        if self.attsd is None:
            self.attsd = 1.0 if len(self.tis) > 1 else 0.1
        if self.artt is None:
            self.artt = self.att - 0.3
        if self.arttsd is None:
            self.arttsd = self.attsd

        # Repeats are supposed to be a list but can be a single number
        if isinstance(self.repeats, int):
            self.repeats = [self.repeats]

        # For now we only support fixed repeats
        if len(self.repeats) == 1:
            # FIXME variable repeats
            self.repeats = self.repeats[0]
        elif len(self.repeats) > 1 and \
            any([ r != self.repeats[0] for r in self.repeats ]):
            raise NotImplementedError("Variable repeats for TIs/PLDs")

        # If we are modelling separate WM/GM components, load the partial volume maps
        if self.pvcorr: 
            self.incwm = True
            self.inferwm = True 

        if self.incwm:
            # FIXME do we need to support fixed numerical PVs?
            self.pvgm = data_model.get_voxel_data(self.pvgm)
            self.pvwm = data_model.get_voxel_data(self.pvwm)
            if (np.array(self.pvgm + self.pvwm) > 1).any():
                raise ValueError("At least one GM and WM PV sum to > 1")

        # Default partition coefficient
        # If we have a separate WM component at each node, self.pc refers to the GM component 
        # only, otherwise it should default to a 'mixed WM/GM' value
        # FIXME same for T1/ATT?
        if self.pc is None: 
            if self.incwm: 
                self.pc = 0.98
            else: 
                self.pc = 0.9

        # With multiple data model structures we need to define nodewise
        # T1/PC/ATT prior etc based on the structure name (WM/GM etc)
        # FIXME In Tom's version the data model defines node slices for the structures
        # start_node = 0
        # for param in ("t1", "pc", "fcalib", "att"):
        #     nodewise_data = np.zeros(self.data_model.n_nodes, dtype=NP_DTYPE)
        #     for structure in self.data_model.model_structures:
        #         end_node = start_node+structure["size"]
        #         if structure["name"] == "WM":
        #             nodewise_data[start_node:end_node] = getattr(self, param + "wm")
        #         else:
        #             nodewise_data[start_node:end_node] = getattr(self, param)
        #         start_node = end_node
        #     setattr(self, param, nodewise_data[:, np.newaxis])
        
        # We probably don't want to use incwm if there are multiple node structures - they
        # probably represent GM/WM nodes and therefore multiple compartments are built into
        # the data model
        if self.incwm and len(self.data_model.model_structure) > 1:
            self.log.warn("Separate GM/WM components defined at each node, but there are also multiple structures in the data model")
            self.log.warn("If the data model includes separate GM/WM nodes you do not need to use --incwm / --pvcorr")

        # Define model parameters
        if self.artonly:
            self.inferart = True

        if not self.artonly:
            # Always include tissue parameters unless specifically arterial only
            self.params = [
                get_parameter("ftiss", dist="Normal", 
                            mean=1.5, prior_var=1e6, post_var=1.5, 
                            post_init=self._init_flow,
                            **options)
            ]
            if self.inferatt: 
                self.params.append(
                    get_parameter("delttiss", dist="Normal", 
                                mean=self.att, var=self.attsd**2,
                                post_init=self._init_delt,
                                **options)
                    )

            if self.inferwm: 
                self.params.append(
                    get_parameter("fwm", dist="Normal", 
                            mean=0.5, prior_var=1e6, post_var=1.5, 
                            post_init=self._init_flow,
                            **options)
                )
                if self.inferatt:
                    self.params.append(
                        get_parameter("deltwm", dist="Normal", 
                                mean=self.attwm, var=self.attsd**2,
                                post_init=self._init_delt,
                                **options)
                    )

        if self.infert1:
            self.params.append(
                get_parameter("t1", mean=self.t1, var=0.01, **options)
            )

            if self.inferwm:
                self.params.append(
                    get_parameter("t1wm", mean=self.t1wm, var=0.01, **options)
                )

        if self.inferart:
            self.leadscale = 0.01
            self.params.append(
                get_parameter("fblood", dist="Normal",
                              mean=0.0, prior_var=1e6, post_var=1.5,
                              post_init=self._init_fblood,
                              prior_type="A",
                              **options)
            )
            if self.inferatt:
                self.params.append(
                    get_parameter("deltblood", dist="Normal", 
                                mean=self.artt, var=self.arttsd**2,
                                post_init=self._init_delt,
                                **options)
                )

    def tpts(self):
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis)*self.repeats, self.data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        # Generate timings volume using the slicedt value
        t = np.zeros(list(self.data_model.shape) + [self.data_model.n_tpts], dtype=np.float32)
        for z in range(self.data_model.shape[2]):
            t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))

        # Apply mask
        t = t[self.data_model.mask_vol > 0]

        # Time points derived from data space need to be transformed into node space.
        t = t.reshape(-1, 1, self.data_model.n_tpts)
        t = self.data_model.voxels_to_nodes_ts(t, pv_sum=False)
        return t.reshape(-1, self.data_model.n_tpts)

    def __str__(self):
        return "ASL resting state model: %s" % __version__

    def evaluate(self, params, tpts):
        """
        Basic PASL/pCASL kinetic model

        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """

        n_params = len(params) if isinstance(params, list) else params.get_shape().as_list()[0]
        if n_params != len(self.params):
            raise ValueError(f"Model set up to infer {len(self.params)} parameters; "
                "this many parameter arrays must be supplied")

        # Extract parameter tensors
        t = tpts
        param_idx = 0
        if not self.artonly:
            ftiss = params[param_idx]
            param_idx += 1

            if self.inferatt:
                delt = params[param_idx]
                param_idx += 1
            else: 
                delt = self.att 
        
            if self.inferwm:
                fwm = params[param_idx]
                param_idx += 1
                
                if self.inferatt:
                    deltwm = params[param_idx]
                    param_idx += 1   
                else: 
                    deltwm = self.attwm 

            else: 
                fwm = self.fwm
                deltwm = self.attwm              
    
        if self.infert1:
            t1 = params[param_idx]
            param_idx += 1
            if self.inferwm: 
                t1wm = params[param_idx]
                param_idx += 1 

        else:
            t1 = self.t1
            t1wm = self.t1wm

        if self.inferart:
            fblood = params[param_idx]
            deltblood = params[param_idx+1]
            param_idx += 2

        # Extra parameters may be required by subclasses, e.g. dispersion parameters
        extra_params = params[param_idx:]

        if not self.artonly:
            signal = self.tissue_signal(t, ftiss, delt, t1, self.pc, self.fcalib, self.pvgm, extra_params)

            if self.incwm: 
                wmsignal = self.tissue_signal(t, fwm, deltwm, t1wm, self.pcwm, self.fcalibwm, self.pvwm, extra_params)
                signal += wmsignal

        else:
            signal = tf.zeros(tf.shape(t), dtype=tf.float32)

        if self.inferart:
            # FIMXE: is this going to work in surface/hybrid mode?
            signal += self.art_signal(t, fblood, deltblood, extra_params)

        return signal

    def tissue_signal(self, t, ftiss, delt, t1, pc, fcalib, pv=1.0, extra_params=[]):
        """
        PASL/pCASL kinetic model for tissue
        """
        if (extra_params != []) and (extra_params.shape[0] > 0): 
            raise NotImplementedError("Extra tissue parameters not set up yet")

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = tf.greater(t, tf.add(self.tau, delt))
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))

        # Rate constants
        t1_app = 1 / (1 / t1 + fcalib / pc)

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = 2 * t1_app * tf.exp(-delt / self.t1b)
            during_bolus_signal = factor * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = factor * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-t / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = factor * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal))
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return pv * ftiss * signal

    def art_signal(self, t, fblood, deltblood, extra_params):
        """
        PASL/pCASL Kinetic model for arterial curve
        
        To avoid problems with the discontinuous gradient at ti=deltblood
        and ti=deltblood+taub, we smooth the transition at these points
        using a Gaussian convolved step function. The sigma value could
        be exposed as a parameter (small value = less smoothing). This is
        similar to the effect of Gaussian dispersion, but can be computed
        without numerical integration
        """
        if self.casl:
            kcblood = 2 * tf.exp(-deltblood / self.t1b)
        else:
            kcblood = 2 * tf.exp(-t / self.t1b)

        # Boolean masks indicating which voxel-timepoints are in the leadin phase
        # and which in the leadout
        leadout = tf.greater(t, tf.add(deltblood, self.tau/2))
        leadin = tf.logical_not(leadout)

        # If deltblood is smaller than the lead in scale, we could 'lose' some
        # of the bolus, so reduce degree of lead in as deltblood -> 0. We
        # don't really need it in this case anyway since there will be no
        # gradient discontinuity
        leadscale = tf.minimum(deltblood, self.leadscale)
        leadin = tf.logical_and(leadin, tf.greater(leadscale, 0))

        # Calculate lead-in and lead-out signals
        leadin_signal = kcblood * 0.5 * (1 + tf.math.erf((t - deltblood) / leadscale))
        leadout_signal = kcblood * 0.5 * (1 + tf.math.erf(-(t - deltblood - self.tau) / self.leadscale))

        # Form final signal from combination of lead in and lead out signals
        signal = tf.zeros(tf.shape(leadin_signal))
        signal = tf.where(leadin, leadin_signal, signal)
        signal = tf.where(leadout, leadout_signal, signal)

        return fblood*signal

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """
        # return f, None 
        if not self.pvcorr:
            f = tf.math.maximum(data.mean(-1).astype(NP_DTYPE), 0.1)
            return f, None
        else:
            # Do a quick edge correction to up-scale signal in edge voxels 
            # Guard against small number division 
            pvsum = self.pvgm + self.pvwm
            edge_data = data / np.maximum(pvsum, 0.3)[:,None]

            # Intialisation for PVEc: assume a CBF ratio of 3:1, 
            # let g = GM PV, w = WM PV = (1 - g), f = raw CBF, 
            # x = WM CBF. Then, wx + 3gx = f => x = 3f / (1 + 2g)
            f = tf.math.maximum(data.mean(-1).astype(NP_DTYPE), 0.1)
            fwm = f / (1 + 2*self.pvgm)
            if _param.name == 'fwm':
                return fwm, None
            else: 
                return 3 * fwm, None 

    def _init_fblood(self, _param, _t, data):
        """
        Initial value for the fblood parameter
        """
        return tf.math.maximum(tf.reduce_max(data, axis=1), 0.1), None

    def _init_delt(self, _param, t, data):
        """
        Initial value for the delttiss parameter
        """
        if self.att_init == "max":
            max_idx = tf.math.argmax(data, axis=1)
            time_max = tf.squeeze(tf.gather(t, max_idx, axis=1, batch_dims=1), axis=-1)

            if _param.name == 'fwm': 
                return (time_max + 0.3 - self.tau, 
                        self.attsd * np.ones_like(time_max))
            else: 
                return (time_max - self.tau, 
                        self.attsd * np.ones_like(time_max))
        # elif self.data_model.is_volumetric:
        #     if _param.name == 'fwm': 
        #         return self.attwm, self.attsd
        #     else: 
        #         return self.att, self.attsd
        # elif self.data_model.is_hybrid: 
        #     att_init = np.ones(self.data_model.n_nodes, dtype=np.float32)
        #     att_init[self.data_model.surf_slicer] = self.att 
        #     att_init[self.data_model.vol_slicer] = self.attwm
        #     return att_init, self.attsd
        else: 
            return tf.fill((self.data_model.n_nodes,), self.att), None
