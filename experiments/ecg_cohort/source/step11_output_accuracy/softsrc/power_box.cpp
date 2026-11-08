// GNU MPFR directed endpoints. The host runtime provides the libraries but not
// the development header; the fallback declares the public LP64 GMP/MPFR ABI.
#include <cmath>
#include <vector>
#if __has_include(<mpfr.h>)
#include <mpfr.h>
#else
#include <gmp.h>
extern "C" {
typedef long mpfr_prec_t; typedef long mpfr_exp_t; typedef int mpfr_sign_t;
typedef struct {mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign; mpfr_exp_t _mpfr_exp; mp_limb_t* _mpfr_d;} __mpfr_struct;
typedef __mpfr_struct mpfr_t[1]; typedef __mpfr_struct* mpfr_ptr;typedef const __mpfr_struct* mpfr_srcptr;
typedef enum {MPFR_RNDN=0,MPFR_RNDZ,MPFR_RNDU,MPFR_RNDD} mpfr_rnd_t;
void mpfr_init2(mpfr_ptr,mpfr_prec_t);void mpfr_clear(mpfr_ptr);
int mpfr_set_d(mpfr_ptr,double,mpfr_rnd_t);int mpfr_set(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_ui_sub(mpfr_ptr,unsigned long,mpfr_srcptr,mpfr_rnd_t);
int mpfr_add(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_pow(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_div(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
double mpfr_get_d(mpfr_srcptr,mpfr_rnd_t);
const char* mpfr_get_version(void);
}
static_assert(sizeof(long)==8 && sizeof(mp_limb_t)==8,"Fallback supports LP64, GMP 64-bit limbs only.");
#endif
namespace {
struct N {mpfr_t v; N(){mpfr_init2(v,160);} ~N(){mpfr_clear(v);} N(const N&)=delete;N&operator=(const N&)=delete;};
void weight(N& out,double t,N& tau,N& omega,mpfr_rnd_t r){mpfr_set_d(out.v,t,r);mpfr_ui_sub(out.v,1,out.v,r);mpfr_add(out.v,out.v,tau.v,r);mpfr_pow(out.v,out.v,omega.v,r);}
}
extern "C" const char* power_version(){return mpfr_get_version();}
extern "C" int power_box(int rows,int width,int nblocks,const int* blocks,const double* l,const double* h,double om,double ta,double* point,double* low,double* high){
if(rows<1||width<1||nblocks<1||!std::isfinite(om)||om<=0||om>128||ta<std::ldexp(1.,-52)||ta>1||!std::isfinite(ta))return 1;
int kmax=0,total=0;for(int b=0;b<nblocks;b++){if(blocks[b]<1||blocks[b]>256)return 2;total+=blocks[b];kmax=std::max(kmax,blocks[b]);}if(total!=width)return 3;
N tau,omega,totalm,denL,denU,ans;mpfr_set_d(tau.v,ta,MPFR_RNDN);mpfr_set_d(omega.v,om,MPFR_RNDN);
std::vector<N> wl(kmax),wh(kmax),wm(kmax);
for(int r=0;r<rows;r++){int o=r*width;
 for(int b=0;b<nblocks;b++){int k=blocks[b];mpfr_set_d(totalm.v,0,MPFR_RNDN);
  for(int j=0;j<k;j++){
   if(!std::isfinite(l[o+j])||!std::isfinite(h[o+j])||l[o+j]>h[o+j])return 4;
   double a=std::max(0.,std::min(1.,l[o+j])),z=std::max(0.,std::min(1.,h[o+j]));
   weight(wl[j],z,tau,omega,MPFR_RNDD);weight(wh[j],a,tau,omega,MPFR_RNDU);
   weight(wm[j],a+(z-a)*0.5,tau,omega,MPFR_RNDN);mpfr_add(totalm.v,totalm.v,wm[j].v,MPFR_RNDN);
  }
  for(int i=0;i<k;i++){
   mpfr_set(denL.v,wl[i].v,MPFR_RNDU);mpfr_set(denU.v,wh[i].v,MPFR_RNDD);
   for(int j=0;j<k;j++)if(i!=j){mpfr_add(denL.v,denL.v,wh[j].v,MPFR_RNDU);mpfr_add(denU.v,denU.v,wl[j].v,MPFR_RNDD);}
   mpfr_div(ans.v,wl[i].v,denL.v,MPFR_RNDD);low[o+i]=std::max(0.,mpfr_get_d(ans.v,MPFR_RNDD));
   mpfr_div(ans.v,wh[i].v,denU.v,MPFR_RNDU);high[o+i]=std::min(1.,mpfr_get_d(ans.v,MPFR_RNDU));
   mpfr_div(ans.v,wm[i].v,totalm.v,MPFR_RNDN);point[o+i]=mpfr_get_d(ans.v,MPFR_RNDN);
  }
  o+=k;
 }
}return 0;}
