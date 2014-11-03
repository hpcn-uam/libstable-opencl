#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #warning "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2 
#endif

#include "includes/opencl_common.h"

cl_precision stable_pdf_g1(cl_precision theta, constant struct stable_info* stable);

constant cl_precision xgk[31] =   /* abscissae of the 61-point kronrod rule */
{
  0.999484410050490637571325895705811,
  0.996893484074649540271630050918695,
  0.991630996870404594858628366109486,
  0.983668123279747209970032581605663,
  0.973116322501126268374693868423707,
  0.960021864968307512216871025581798,
  0.944374444748559979415831324037439,
  0.926200047429274325879324277080474,
  0.905573307699907798546522558925958,
  0.882560535792052681543116462530226,
  0.857205233546061098958658510658944,
  0.829565762382768397442898119732502,
  0.799727835821839083013668942322683,
  0.767777432104826194917977340974503,
  0.733790062453226804726171131369528,
  0.697850494793315796932292388026640,
  0.660061064126626961370053668149271,
  0.620526182989242861140477556431189,
  0.579345235826361691756024932172540,
  0.536624148142019899264169793311073,
  0.492480467861778574993693061207709,
  0.447033769538089176780609900322854,
  0.400401254830394392535476211542661,
  0.352704725530878113471037207089374,
  0.304073202273625077372677107199257,
  0.254636926167889846439805129817805,
  0.204525116682309891438957671002025,
  0.153869913608583546963794672743256,
  0.102806937966737030147096751318001,
  0.051471842555317695833025213166723,
  0.000000000000000000000000000000000
};

/* xgk[1], xgk[3], ... abscissae of the 30-point gauss rule. 
   xgk[0], xgk[2], ... abscissae to optimally extend the 30-point gauss rule */

constant cl_precision wg[15] =    /* weights of the 30-point gauss rule */
{
  0.007968192496166605615465883474674,
  0.018466468311090959142302131912047,
  0.028784707883323369349719179611292,
  0.038799192569627049596801936446348,
  0.048402672830594052902938140422808,
  0.057493156217619066481721689402056,
  0.065974229882180495128128515115962,
  0.073755974737705206268243850022191,
  0.080755895229420215354694938460530,
  0.086899787201082979802387530715126,
  0.092122522237786128717632707087619,
  0.096368737174644259639468626351810,
  0.099593420586795267062780282103569,
  0.101762389748405504596428952168554,
  0.102852652893558840341285636705415
};

constant cl_precision wgk[31] =   /* weights of the 61-point kronrod rule */
{
  0.001389013698677007624551591226760,
  0.003890461127099884051267201844516,
  0.006630703915931292173319826369750,
  0.009273279659517763428441146892024,
  0.011823015253496341742232898853251,
  0.014369729507045804812451432443580,
  0.016920889189053272627572289420322,
  0.019414141193942381173408951050128,
  0.021828035821609192297167485738339,
  0.024191162078080601365686370725232,
  0.026509954882333101610601709335075,
  0.028754048765041292843978785354334,
  0.030907257562387762472884252943092,
  0.032981447057483726031814191016854,
  0.034979338028060024137499670731468,
  0.036882364651821229223911065617136,
  0.038678945624727592950348651532281,
  0.040374538951535959111995279752468,
  0.041969810215164246147147541285970,
  0.043452539701356069316831728117073,
  0.044814800133162663192355551616723,
  0.046059238271006988116271735559374,
  0.047185546569299153945261478181099,
  0.048185861757087129140779492298305,
  0.049055434555029778887528165367238,
  0.049795683427074206357811569379942,
  0.050405921402782346840893085653585,
  0.050881795898749606492297473049805,
  0.051221547849258772170656282604944,
  0.051426128537459025933862879215781,
  0.051494729429451567558340433647099
};

cl_precision stable_pdf_alpha_neq1(cl_precision theta, constant struct stable_info *args)
{
    cl_precision g, cos_theta, aux, V;

    //  g   = dist->beta_;
    //  aux = theta+dist->theta0_;
    //  V   = M_PI_2-theta;

    //  if ((g==1 && aux < THETA_TH*1.1 && dist->alfa <1) || (g==-1 && V<THETA_TH*1.1 && dist->alfa>1)) {
    //    V = dist->Vbeta1;// printf("");
    // }
    //  else {

    cos_theta = cos(theta);
    aux = (args->theta0_ + theta) * args->alfa;
    V = log(cos_theta / sin(aux)) * args->alfainvalfa1 +
        + log(cos(aux - theta) / cos_theta) + args->k1;
    //  }

#ifdef DEBUG
    integ_eval++;
#endif

    g = V + args->xxipow;
    //g>6.55 -> exp(g-exp(g)) < 2.1E-301
    if (g > 6.55 || g < -700) return 0.0;
    //Taylor: x·exp(-x) ~ x·(1-x) cuando x ~ 0
    //Si g < 1.52e-8 -> g·exp(-g) = g·(1-g) con precision cl_precision.
    //Asi nos ahorramos calcular una exponencial (que es costoso).
    else  g = exp(g);
    //  if(isnan(g) || isinf(g)) {return 0.0;}
    //  else if (g < 1.522e-8) {return (1.0-g)*g;}
    /*  else*/ g = exp(-g) * g;
    if (isnan(g) || isinf(g) || g < 0)
    {
        return 0.0;
    }
    /*
      fprintf(FINTEG,"%1.16lf\t%1.16lf\t%1.16lf\t%1.16e\n",
              args->alfa,args->beta_,theta,g);
    */

    return g;
}

cl_precision stable_pdf_alpha_eq1(cl_precision theta, constant struct stable_info* stable)
{ 
    cl_precision g, V, aux;

    //  g   = dist->beta_;
    //  aux = theta+dist->theta0_;
    //  V   = M_PI_2-theta;

    //  if ((g==1 && aux < THETA_TH*1.1) || (g==-1 && V<THETA_TH*1.1)) {
    //    V = dist->Vbeta1;// printf("");
    //  }
    //  else {
    aux = (stable->beta_ * theta + M_PI_2) / cos(theta);
    V = sin(theta) * aux / stable->beta_ + log(aux) + stable->k1;
    //  }

    g = V + stable->xxipow;
    //Obtenemos log(g), en realidad
    //Taylor: exp(-x) ~ 1-x en x ~ 0
    //Si g<1.52e-8 -> exp(-g)=(1-g) -> g·exp(-g) = g·(1-g) con precision cl_precision.
    //Asi nos ahorramos calcular una exponencial. (que es costoso).
    if(isnan(g)) return 0.0;
    if ((g = exp(g)) < 1.522e-8 ) return (1.0 - g) * g;
    g = exp(-g) * g;
    if (isnan(g) || g < 0) return 0.0;

    return g;
}

kernel void stable_pdf(global cl_precision* gauss, global cl_precision* kronrod, constant struct stable_info* stable)
{
    size_t thread_id = get_global_id(0);
    size_t subinterval_index = get_local_id(0);
    size_t interval = get_group_id(0);

    const int gauss_eval_points = GK_POINTS / 4;
    const int kronrod_eval_points = GK_POINTS / 2 + 1;
    local cl_precision gauss_sum[GK_POINTS / 4];
    local cl_precision kronrod_sum[GK_POINTS / 2 + 1];

    if(subinterval_index < kronrod_eval_points)
    {
      const cl_precision center = stable->ibegin + stable->subinterval_length * interval + stable->half_subint_length;
      const cl_precision abscissa = stable->half_subint_length * xgk[subinterval_index]; // Translated integrand evaluation
      cl_precision fval1, fval2, fsum;

      if(stable->integrand == PDF_ALPHA_EQ1) 
      {
        fval1 = stable_pdf_alpha_eq1(center - abscissa, stable);
        fval2 = stable_pdf_alpha_eq1(center + abscissa, stable);
      }
      else
      {
        fval1 = stable_pdf_alpha_neq1(center - abscissa, stable);
        fval2 = stable_pdf_alpha_neq1(center + abscissa, stable);
      }

      fsum = fval1 + fval2;

      if(subinterval_index == kronrod_eval_points - 1)
        fsum /= 2;

      if(subinterval_index % 2 == 1)
        gauss_sum[subinterval_index / 2] = wg[subinterval_index / 2] * fsum;

      kronrod_sum[subinterval_index] = wgk[subinterval_index] * fsum; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(subinterval_index == kronrod_eval_points)
    {
      int i;
      double loc_gauss = 0, loc_kronrod = 0;

      for(i = 0; i < 15; i++)
      {
        loc_gauss += gauss_sum[i];
        loc_kronrod += kronrod_sum[i];
      }

      for(; i < 31; i++)
      {
        loc_kronrod += kronrod_sum[i];
      }

      gauss[interval] = loc_gauss * stable->half_subint_length;
      kronrod[interval] = loc_kronrod * stable->half_subint_length;
    }
}
