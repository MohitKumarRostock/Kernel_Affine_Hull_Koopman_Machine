// Forward interval evaluator. Binary32/binary64 input data are exact dyadics.
// GNU MPFI 1.5 and GNU MPFR 4 supply all rounded arithmetic. No BLAS/fast-math.
#include <cmath>
#include <vector>
#include <algorithm>
#include <new>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <gmp.h>
#if __has_include(<mpfi.h>)
#include <mpfi.h>
#else
// Public LP64 MPFR/MPFI ABI subset for runtimes shipping the libraries only.
extern "C" {
typedef long mpfr_prec_t; typedef long mpfr_exp_t; typedef int mpfr_sign_t;
typedef struct {mpfr_prec_t _mpfr_prec;mpfr_sign_t _mpfr_sign;mpfr_exp_t _mpfr_exp;mp_limb_t *_mpfr_d;} __mpfr_struct;
typedef enum {MPFR_RNDN=0, MPFR_RNDZ=1, MPFR_RNDU=2, MPFR_RNDD=3} mpfr_rnd_t;
typedef struct {__mpfr_struct left,right;} __mpfi_struct;
typedef __mpfi_struct mpfi_t[1];
void mpfi_init2(__mpfi_struct*,mpfr_prec_t);void mpfi_clear(__mpfi_struct*);
int mpfi_set_d(__mpfi_struct*,double);int mpfi_set(__mpfi_struct*,const __mpfi_struct*);
int mpfi_set_si(__mpfi_struct*,long);int mpfi_const_pi(__mpfi_struct*);
const char* mpfi_get_version(void);const char* mpfr_get_version(void);
int mpfi_bounded_p(const __mpfi_struct*);int mpfi_nan_p(const __mpfi_struct*);int mpfi_has_zero(const __mpfi_struct*);
int mpfr_set(__mpfr_struct*,const __mpfr_struct*,mpfr_rnd_t);
int mpfr_cmp(const __mpfr_struct*,const __mpfr_struct*);
int mpfr_cmp_si(const __mpfr_struct*,long);
double mpfr_get_d(const __mpfr_struct*,mpfr_rnd_t);
#define BIN(s) int mpfi_##s(__mpfi_struct*,const __mpfi_struct*,const __mpfi_struct*);
BIN(add) BIN(sub) BIN(mul) BIN(div) BIN(intersect)
#define UN(s) int mpfi_##s(__mpfi_struct*,const __mpfi_struct*);
UN(neg) UN(abs) UN(acos) UN(sqrt) UN(sqr) UN(exp) UN(expm1) UN(log)
int mpfi_div_ui(__mpfi_struct*,const __mpfi_struct*,unsigned long);
int mpfi_mul_d(__mpfi_struct*,const __mpfi_struct*,double);
int mpfi_add_d(__mpfi_struct*,const __mpfi_struct*,double);
int mpfi_interv_d(__mpfi_struct*,double,double);
}
static_assert(sizeof(long)==8 && sizeof(mp_limb_t)==8 && sizeof(__mpfr_struct)==32, "Unsupported MPFR ABI");
#endif
static thread_local unsigned prec=80;
struct I {mpfi_t a;
  I(){mpfi_init2(a,prec);mpfi_set_si(a,0);} explicit I(double d){mpfi_init2(a,prec);mpfi_set_d(a,d);}
  I(const I& z){mpfi_init2(a,prec);mpfi_set(a,z.a);} I& operator=(const I& z){if(this!=&z)mpfi_set(a,z.a);return *this;} ~I(){mpfi_clear(a);}
};
using V=std::vector<I>;
static bool valid(const I& x){return !mpfi_nan_p(x.a)&&mpfi_bounded_p(x.a)&&mpfr_cmp(&x.a->left,&x.a->right)<=0;}
static bool zero(const I& x){return mpfr_cmp_si(&x.a->left,0)==0 && mpfr_cmp_si(&x.a->right,0)==0;}
static void addprod(I& res,const I& x,const I& y,I& t){mpfi_mul(t.a,x.a,y.a);mpfi_add(res.a,res.a,t.a);}
static void addprodd(I& res,const I& x,double y,I& t){mpfi_mul_d(t.a,x.a,y);mpfi_add(res.a,res.a,t.a);}
static void norm2(I& r,const V& v,I& t){mpfi_set_si(r.a,0);for(auto &x:v){mpfi_sqr(t.a,x.a);mpfi_add(r.a,r.a,t.a);}}
static double low(const I& v){return mpfr_get_d(&v.a->left,MPFR_RNDD);}
static double high(const I& v){return mpfr_get_d(&v.a->right,MPFR_RNDU);}
static void ivmin(I& v,const I& x){if(mpfr_cmp(&x.a->left,&v.a->left)<0)mpfr_set(&v.a->left,&x.a->left,MPFR_RNDD);if(mpfr_cmp(&x.a->right,&v.a->right)<0)mpfr_set(&v.a->right,&x.a->right,MPFR_RNDU);}
// u is either x*scale+offset (mode 0) or (x-offset)/scale (mode 1).
struct Enc{
 int d,q,n,mode; V scale,offset,slopes,c0,scoef,ncoef;
 Enc(int dd,int qq,int nn,int mm,const double* P,const double* A,const double* S,const double* B,const double* Y,const double* sc,const double* of):
 d(dd),q(qq),n(nn),mode(mm),scale(d),offset(d),slopes(n*d),c0(n),scoef(n),ncoef(n*d) {
  I t,t2,rr,rr0; V SA(q*n), SP(q*d),diff(q);
  for(int l=0;l<d;l++){mpfi_set_d(scale[l].a,sc[l]);mpfi_set_d(offset[l].a,of[l]);}
  for(int i=0;i<q;i++)for(int h=0;h<q;h++){
    I sh(S[i*q+h]);
    for(int j=0;j<n;j++)addprodd(SA[i*n+j],sh,A[h*n+j],t);
    for(int l=0;l<d;l++)addprodd(SP[i*d+l],sh,P[l*q+h],t);
  }
  for(int i=0;i<q;i++){mpfi_sqr(t.a,SA[i*n].a);mpfi_add(rr0.a,rr0.a,t.a);}
  for(int j=0;j<n;j++){
   if(j){mpfi_set_si(rr.a,0);for(int i=0;i<q;i++){
     mpfi_sub(diff[i].a,SA[i*n+j].a,SA[i*n].a);
     mpfi_sqr(t.a,SA[i*n+j].a);mpfi_add(rr.a,rr.a,t.a);
     for(int l=0;l<d;l++)addprod(slopes[j*d+l],SP[i*d+l],diff[i],t);
   }mpfi_sub(c0[j].a,rr0.a,rr.a);mpfi_div_ui(c0[j].a,c0[j].a,2*q);
   for(int l=0;l<d;l++)mpfi_div_ui(slopes[j*d+l].a,slopes[j*d+l].a,q);}
   for(int k=0;k<n;k++){I b(B[j*n+k]);mpfi_add(scoef[j].a,scoef[j].a,b.a);for(int l=0;l<d;l++)addprodd(ncoef[j*d+l],b,Y[l*n+k],t);}
  }
 }
 int run(const double* x,I& score,const I& pi,const I& csbox,const I& unit){
  V u(d),v(d),res(d),rk(n); I t,t2,s,sabs,rr,rn,un,dot,cs,ang,rad;
  for(int l=0;l<d;l++){mpfi_set_d(u[l].a,x[l]);if(mode==0){mpfi_mul(u[l].a,u[l].a,scale[l].a);mpfi_add(u[l].a,u[l].a,offset[l].a);}
    else{mpfi_sub(u[l].a,u[l].a,offset[l].a);mpfi_div(u[l].a,u[l].a,scale[l].a);}}
  int pivot=0;for(int j=0;j<n;j++){rk[j]=c0[j];for(int l=0;l<d;l++)addprod(rk[j],slopes[j*d+l],u[l],t);
   if(!valid(rk[j]))return 4;
   if(mpfr_cmp(&rk[j].a->right,&rk[pivot].a->right)>0)pivot=j;
  }
  // Subtract a single certified logit, not separately rounded endpoint maxima.
  I ref(rk[pivot]);for(int j=0;j<n;j++){
   if(j==pivot)mpfi_set_si(rk[j].a,1);else{mpfi_sub(rk[j].a,rk[j].a,ref.a);mpfi_exp(rk[j].a,rk[j].a);}
   addprod(s,scoef[j],rk[j],t);
   for(int l=0;l<d;l++)addprod(v[l],ncoef[j*d+l],rk[j],t);
  }
  if(!valid(s))return 4;
  if(mpfi_has_zero(s.a))return zero(s)?2:1;
  for(int l=0;l<d;l++){mpfi_mul(t.a,s.a,u[l].a);mpfi_sub(res[l].a,v[l].a,t.a);addprod(dot,v[l],u[l],t);}
  norm2(rn,v,t);norm2(un,u,t);norm2(rr,res,t);if(!valid(rn)||!valid(un)||!valid(rr))return 4;
  if(zero(rn)||zero(un))mpfi_set_si(cs.a,0); // exact native zero-norm convention
  else{
    if(mpfi_has_zero(rn.a)||mpfi_has_zero(un.a))return 3;
    mpfi_mul(t.a,rn.a,un.a);mpfi_sqrt(t.a,t.a);if(!valid(t)||mpfi_has_zero(t.a))return 3;
    mpfi_div(cs.a,dot.a,t.a);if(mpfr_cmp_si(&s.a->right,0)<0)mpfi_neg(cs.a,cs.a);
    mpfi_intersect(cs.a,cs.a,csbox.a);if(!valid(cs))return 4;
  }
  mpfi_acos(ang.a,cs.a);mpfi_div(ang.a,ang.a,pi.a);
  mpfi_sqrt(rr.a,rr.a);mpfi_abs(sabs.a,s.a);mpfi_div(rr.a,rr.a,sabs.a);if(!valid(rr))return 4;
  mpfi_neg(rr.a,rr.a);mpfi_expm1(rad.a,rr.a);mpfi_neg(rad.a,rad.a);
  mpfi_sqr(t.a,ang.a);mpfi_sqr(t2.a,rad.a);mpfi_add(t.a,t.a,t2.a);mpfi_div_ui(t.a,t.a,2);mpfi_sqrt(score.a,t.a);
  mpfi_intersect(score.a,score.a,unit.a);return valid(score)?0:4;
 }
};
struct Bank{int bits;std::vector<Enc> enc;I pi,csbox,unit;
 Bank(int p):bits(p){mpfi_const_pi(pi.a);mpfi_interv_d(csbox.a,-1,1);mpfi_interv_d(unit.a,0,1);}
};
extern "C" {
 const char* k11_mpfi(){return mpfi_get_version();}const char* k11_mpfr(){return mpfr_get_version();}
 void* k11_new(int p){try{prec=p;return new Bank(p);}catch(...){return nullptr;}}
 void k11_free(void* b){delete (Bank*)b;}
 int k11_add(void* b,int d,int q,int n,int mode,const double* P,const double* A,const double* S,const double* B,const double* Y,const double* sc,const double* of){try{
  auto& bank=*(Bank*)b;prec=bank.bits;bank.enc.emplace_back(d,q,n,mode,P,A,S,B,Y,sc,of);return 0;
 }catch(...){return -1;}}
 // grp maps encoders to flattened states. Q concatenates block matrices (row-major).
 int k11_eval(void* b,int rows,int d,int ns,int nb,const int* grp,const int* blocks,const double* X,const double* Q,double tau,
  double* elo,double* ehi,int* est,double* slo,double* shi,double* flo,double* fhi){try{
  auto& bank=*(Bank*)b;prec=bank.bits;int ne=bank.enc.size();
  I t,g,tf;V ec(ne),sc(ns),ls(ns);
  for(int r=0;r<rows;r++){
   std::vector<int> seen(ns,0),fail(ns,0);
   for(int e=0;e<ne;e++){
     int f=bank.enc[e].run(X+r*d,ec[e],bank.pi,bank.csbox,bank.unit);
     est[r*ne+e]=f;int j=grp[e];if(f){fail[j]=1;continue;}
     elo[r*ne+e]=low(ec[e]);ehi[r*ne+e]=high(ec[e]);if(!seen[j])sc[j]=ec[e];else ivmin(sc[j],ec[e]);seen[j]=1;
   }
   int sb=0,qb=0;
   for(int bi=0;bi<nb;bi++){int sz=blocks[bi];bool bad=false;
     for(int k=0;k<sz;k++){int j=sb+k;if(fail[j]||!seen[j]){bad=true;continue;}
       slo[r*ns+j]=low(sc[j]);shi[r*ns+j]=high(sc[j]);
       mpfi_neg(ls[j].a,sc[j].a);mpfi_add_d(ls[j].a,ls[j].a,1.);mpfi_add_d(ls[j].a,ls[j].a,tau);mpfi_log(ls[j].a,ls[j].a);
     }
     if(!bad && !Q){
       I partial, denom;
       for(int k=0;k<sz;k++){
         if(k>0){mpfi_mul_d(t.a,ls[sb+k].a,double(k));mpfi_sub(g.a,partial.a,t.a);mpfi_set_si(denom.a,long(k)*(k+1));mpfi_sqrt(denom.a,denom.a);mpfi_div(g.a,g.a,denom.a);if(valid(g)){flo[r*ns+sb+k-1]=low(g);fhi[r*ns+sb+k-1]=high(g);}}
         mpfi_add(partial.a,partial.a,ls[sb+k].a);
       }
       mpfi_set_si(denom.a,sz);mpfi_sqrt(denom.a,denom.a);mpfi_div(g.a,partial.a,denom.a);if(valid(g)){flo[r*ns+sb+sz-1]=low(g);fhi[r*ns+sb+sz-1]=high(g);}
     }else if(!bad)for(int k=0;k<sz;k++){
       mpfi_set_si(g.a,0);for(int j=0;j<sz;j++)addprodd(g,ls[sb+j],Q[qb+j*sz+k],t);
       if(valid(g)){flo[r*ns+sb+k]=low(g);fhi[r*ns+sb+k]=high(g);}
     }
     sb+=sz;qb+=sz*sz;
   }
  }
  return 0;
 }catch(...){return -1;}}
}
