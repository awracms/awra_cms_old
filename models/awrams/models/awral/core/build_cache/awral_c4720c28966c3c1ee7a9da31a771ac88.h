typedef struct {
    double *restrict s0, *restrict ss, *restrict sd, *restrict mleaf;
} HRUState;

typedef struct {
    double *restrict sg, *restrict sr;
    HRUState hru[2];
} States;

typedef struct {
    const double *ne, *height, *hypsperc;
} Hypsometry;

typedef struct {
    const double alb_dry;
    const double alb_wet;
    const double cgsmax;
    const double er_frac_ref;
    const double fsoilemax;
    const double lairef;
    const double rd;
    const double s_sls;
    const double sla;
    const double tgrow;
    const double tsenc;
    const double ud0;
    const double us0;
    const double vc;
    const double w0lime;
    const double w0ref_alb;
    const double wdlimu;
    const double wslimu;
} HRUParameters;

typedef struct {
    const double kr_coeff;
    const double pair;
    const double slope_coeff;
} Parameters;

typedef struct {
    const double * pt;
    const double * tat;
    const double * radcskyt;
    const double * rgt;
    const double * u2t;
    const double * avpt;
} Forcing;

typedef struct {
    const double * kr_sd;
    const double * kssat;
    const double * k0sat;
    const double * prefr;
    const double * k_gw;
    const double * kdsat;
    const double * ssmax;
    const double * kr_0s;
    const double * slope;
    const double * s0max;
    const double * k_rout;
    const double * sdmax;
    const double * ne;
} Spatial;

typedef struct {
    const double * fhru;
    const double * hveg;
    const double * laimax;
} HRUSpatial;

typedef struct {
    double *restrict s0;
    double *restrict ss;
    double *restrict sd;
    double *restrict mleaf;
} HRUOutputs;

typedef struct {
    double *restrict e0;
    double *restrict etot;
    double *restrict dd;
    double *restrict s0;
    double *restrict ss;
    double *restrict sd;
    double *restrict qtot;
    double *restrict sr;
    double *restrict sg;
    HRUOutputs hru[2];
} Outputs;


void awral(Forcing inputs, Outputs outputs, States initial_states, States final_states, 
           Parameters params, Spatial spatial, Hypsometry hypso, HRUParameters *hruparams, HRUSpatial *hruspatial,
           int timesteps, int cells);


