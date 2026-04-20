# Aim

Build a numerical post-processing workflow to quantify how the spatial variation of the hydraulic-gradient magnitude \(|\nabla h(x,y)|\) changes the fastest path (FP) relative to the least-resistance path (LRP).

- **FP** uses the true slowness field
  \[
  m(x,y)=\frac{1}{|v(x,y)|}=\frac{1}{K(x,y)\,|\nabla h(x,y)|}.
  \]
- **LRP** uses the approximate slowness field
  \[
  m_0(x,y)=\frac{1}{K(x,y)\,\bar g},
  \]
  where \(\bar g\) is a chosen constant approximation to \(|\nabla h|\).

The goal is to compute, **along both FP and LRP**, the geometric curvature and the normal derivatives of the log-fields, and then compare their balance laws.

---

# Key numerical formulae

## 1. Hydraulic and velocity fields

Solve
\[
\nabla\cdot(K\nabla h)=0
\]
with constant head drop left-to-right and no-flow top/bottom boundaries.

Then compute
\[
\nabla h=(h_x,h_y),
\qquad
|\nabla h|=\sqrt{h_x^2+h_y^2},
\]
\[
\mathbf v=-K\nabla h,
\qquad
|v|=K|\nabla h|.
\]

## 2. Slowness fields

True FP slowness:
\[
m=\frac{1}{|v|}=\frac{1}{K|\nabla h|}.
\]

Approximate LRP slowness:
\[
m_0=\frac{1}{K\bar g}.
\]

## 3. Path representation

For either path, use an ordered list of points
\[
\gamma(s)=(x(s),y(s))
\]
reparameterized by arclength \(s\).

## 4. Tangent, normal, curvature

Compute
\[
\mathbf t=(x_s,y_s),
\qquad
\mathbf N=(-y_s,x_s)
\]
with consistent sign convention.

Signed curvature:
\[
\kappa=x_s y_{ss}-y_s x_{ss}.
\]

Discrete centered-difference version for nearly uniform \(\Delta s\):
\[
x_s(i)\approx \frac{x_{i+1}-x_{i-1}}{2\Delta s},
\qquad
y_s(i)\approx \frac{y_{i+1}-y_{i-1}}{2\Delta s},
\]
\[
x_{ss}(i)\approx \frac{x_{i+1}-2x_i+x_{i-1}}{\Delta s^2},
\qquad
y_{ss}(i)\approx \frac{y_{i+1}-2y_i+y_{i-1}}{\Delta s^2},
\]
\[
\kappa_i=x_s(i)y_{ss}(i)-y_s(i)x_{ss}(i).
\]

## 5. Normal derivative of a scalar field

For any scalar field \(f(x,y)\),
\[
\partial_N f=\nabla f\cdot \mathbf N=f_x N_x+f_y N_y.
\]

In practice, compute the grid fields
\[
\ln K,
\qquad
\ln |\nabla h|,
\qquad
\ln |v|,
\]
then compute their spatial gradients and interpolate those gradients to the path points.

## 6. Required quantities along FP and LRP

Compute along each path:
\[
\kappa(s),
\qquad
\partial_N \ln |v|(s),
\qquad
\partial_N \ln K(s),
\qquad
\partial_N \ln |\nabla h|(s).
\]

Identity check:
\[
\partial_N \ln |v|=\partial_N \ln K+\partial_N \ln |\nabla h|.
\]

## 7. Curvature balance laws

For the true FP in the true medium:
\[
\kappa_{\mathrm{FP}}\approx -\partial_N\ln |v|.
\]

Since
\[
|v|=K|\nabla h|,
\]
this becomes
\[
\kappa_{\mathrm{FP}}
\approx
-\partial_N\ln K
-\partial_N\ln |\nabla h|.
\]

For the LRP approximation:
\[
\kappa_{\mathrm{LRP}}\approx -\partial_N\ln K.
\]

So the neglected bending contribution is
\[
\Delta \kappa
\approx
-\partial_N\ln |\nabla h|.
\]

## 8. Residuals / diagnostics

True-FP residual:
\[
R_{\mathrm{FP}}(s)=\kappa_{\mathrm{FP}}(s)+\partial_N\ln |v|(s).
\]

LRP residual evaluated in the true medium:
\[
R_{\mathrm{LRP}}(s)=\kappa_{\mathrm{LRP}}(s)+\partial_N\ln |v|(s).
\]

If signs are consistent, \(R_{\mathrm{FP}}\) should be small, while \(R_{\mathrm{LRP}}\) measures the non-stationarity of LRP in the true medium.

## 9. Minimal algorithm summary

1. Solve for \(h\) from \(\nabla\cdot(K\nabla h)=0\).
2. Compute \(|\nabla h|\), \(|v|\), \(m\), and \(m_0\).
3. Compute FP from \(m\) and LRP from \(m_0\).
4. Resample each path by arclength.
5. Compute \(\mathbf t\), \(\mathbf N\), and \(\kappa\) along each path.
6. Compute grid gradients of \(\ln K\), \(\ln|\nabla h|\), and \(\ln|v|\).
7. Interpolate those gradients to the path points.
8. Compute normal derivatives by dotting with \(\mathbf N\).
9. Compare curvature and normal-derivative balances along FP and LRP.
