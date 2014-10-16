#define GK_POINTS 61
#define GK_SUBDIVISIONS 4


struct stable_info {
    double theta;
    double beta_;
    double k1;
    double xxipow;
    double ibegin;
    double iend;
    double subinterval_length;
    double half_subint_length;
    unsigned int threads_per_interval;
    unsigned int gauss_points;
    unsigned int kronrod_points;
};
