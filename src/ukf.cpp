#include "ukf.h"
#include "Eigen/Dense"
#include <cmath>
#include <iostream>

// #define DEBUG 1
#define NORMALIZE_ANGLE

// I am trying to combine both lidar and radar measurements in to a single,
// however it seems the miss of rdot fail causes the prediction from measurement
// fail to converge observation vector

// #define SENSOR_MIX

#define UPDATE_TIME_INTERVAL 10 // mircoseconds

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during
  // init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during
  // init)
  use_radar_ = true;

  is_initialized_ = false;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // std_a_ = 1.5;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 8;

  // std_yawdd_ = 0.8;
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member
   * properties. Hint: one or more values initialized above might be wildly
   * off...
   */

  time_us_ = 0;
  // State dimension
  n_x_ = 5;

  // measuremnent dimension
  n_z_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  S_ = MatrixXd(n_x_, n_x_);

  R_ = MatrixXd(n_z_, n_z_);

  R_radar_ = MatrixXd(3, 3);

  R_lidar_ = MatrixXd(2, 2);

  R_v_ = MatrixXd(1, 1);

  z_ = VectorXd(n_z_);

  meas_t_ = VectorXd(3);

  z_pred_ = VectorXd(n_z_);

  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // clang-format off
  VectorXd std_radar(3),std_lidar(2) ;

  std_radar << std_radr_,std_radphi_,std_radrd_;
  std_lidar << std_laspx_,std_laspy_ ;

  R_radar_ = (std_radar.array() * std_radar.array()).matrix().asDiagonal();
  R_lidar_ = (std_lidar.array() * std_lidar.array()).matrix().asDiagonal();
  R_v_(0,0) = std_a_  ;
  // clang-format on

  R_.fill(0.0);

  R_.topLeftCorner(2, 2) = R_lidar_;

  R_.block(2, 2, 3, 3) = R_radar_;
  if (n_z_ > 5) {
    R_.bottomRightCorner(1, 1) = R_v_;
  }
}

UKF::~UKF() {}

double normalize(double x) {
  double theta = fmod(x, 2. * M_PI);

  // theta = theta + floor((M_PI - theta) / 2. * M_PI) * 2. * M_PI;
  return theta;
}
// TODO : Augmented SIGMA Point Generation
void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented process noise covariance matrix
  MatrixXd cov_aug = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);

  cov_aug.fill(0.0);

  cov_aug(0, 0) = std_a_ * std_a_;
  cov_aug(1, 1) = std_yawdd_ * std_yawdd_;

  // create augmented sigma points
  P_aug.fill(0.0);

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, n_aug_ - n_x_, n_aug_ - n_x_) = cov_aug;

  // create augmented square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  double gamma = sqrt(lambda_ + n_aug_);

  x_aug.fill(0.0);

  x_aug.block(0, 0, n_x_, 1) = x_;

  Xsig_aug.fill(0.0);
  Xsig_aug.colwise() += x_aug;

  Xsig_aug.block(0, 1, n_aug_, n_aug_) += gamma * A;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) -= gamma * A;

  // print result
#ifdef DEBUG
  cout << "Xsig_aug = " << endl << Xsig_aug << endl;
#endif

  // write result
  *Xsig_out = Xsig_aug;
}

// TODO : SIGMA Point Prediction
/**
 */

void UKF::SigmaPointPrediction(MatrixXd *Xsig_out) {

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  double delta_t = delta_t_; // time diff in sec
  Xsig_pred.fill(0.0);
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  // print result
#ifdef DEBUG
  cout << "Xsig_pred = " << endl << Xsig_pred << endl;
#endif

  // write result
  *Xsig_out = Xsig_pred;
}

// TODO : Predict Mean and Covariance Matrix

void UKF::PredictMeanAndCovariance(VectorXd *x_out, MatrixXd *P_out) {
  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // set weights

  // predict state mean
  x.setZero();
  x = Xsig_pred_ * weights_;

  // normalized angle
  P.setZero();

  MatrixXd x_diff = Xsig_pred_.colwise() - x;

  // state difference
  // VectorXd x_diff = Xsig_pred_.col(i) - x;
#ifdef NORMALIZE_ANGLE
  x_diff.row(3).unaryExpr([](double x) { return normalize(x); });
#endif

  P = (x_diff.array().rowwise() * weights_.transpose().array()).matrix() *
      x_diff.transpose();

  // print result
#ifdef DEBUG
  cout << "Predicted state" << endl;
  cout << x << endl;
  cout << "Predicted covariance matrix" << endl;
  cout << P << endl;
#endif

  // write result
  *x_out = x;
  *P_out = P;
}

void UKF::PredictMeasurement(VectorXd *z_out, MatrixXd *S_out,
                             MatrixXd *Zsig_out) {

  // create example matrix with predicted sigma points
  MatrixXd Xsig_pred = Xsig_pred_;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  Zsig.fill(0.0);
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model

    if (n_z_ == 2) {
      Zsig(0, i) = p_x; // x
      Zsig(1, i) = p_y; // y
    }

    if (n_z_ == 3) {
      Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
      Zsig(1, i) = atan2(p_y, p_x);                                     // phi
      Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
    }

    if (n_z_ >= 5) {
      Zsig(0, i) = p_x;                         // x
      Zsig(1, i) = p_y;                         // y
      Zsig(2, i) = sqrt(p_x * p_x + p_y * p_y); // r
      Zsig(3, i) = atan2(p_y, p_x);             //      phi
      Zsig(4, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
      if (n_z_ > 5)
        Zsig(5, i) = v; // v
    }
  }

  // mean predicted measurement
  z_pred.fill(0.0);

  z_pred = Zsig * weights_;

#ifdef NORMALIZE_ANGLE
  if (n_z_ == 3) {
    z_pred(1) = normalize(z_pred(1));
  }
  if (n_z_ >= 5) {
    z_pred(3) = normalize(z_pred(3));
  }
#endif

  // innovation covariance matrix S
  S.fill(0.0);
  MatrixXd z_diff = Zsig.colwise() - z_pred;

#ifdef NORMALIZE_ANGLE
  if (n_z_ == 3) {

    z_diff.row(1).unaryExpr([](double x) { return normalize(x); });
  }
  if (n_z_ >= 5) {
    z_diff.row(3).unaryExpr([](double x) { return normalize(x); });
  }
  // apply noarmalization
#endif

  S = (z_diff.array().rowwise() * weights_.transpose().array()).matrix() *
      z_diff.transpose();

  // add measurement noise covariance matrix
  if (n_z_ == 2) {
    S = S + R_lidar_;
  }

  if (n_z_ == 3) {
    S = S + R_radar_;
  }

  if (n_z_ >= 5) {
    S = S + R_;
  }
// print result
#ifdef DEBUG
  cout << "z_pred: " << endl << z_pred << endl;
  cout << "S: " << endl << S << endl;
#endif
  // write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::UpdateState(VectorXd *x_out, MatrixXd *P_out) {

  MatrixXd Xsig_pred = Xsig_pred_;
  MatrixXd P = P_;
  MatrixXd S = S_;
  MatrixXd Zsig = Zsig_;
  VectorXd x = x_;
  VectorXd z = z_;
  VectorXd z_pred = z_pred_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  MatrixXd xdiff = (Xsig_pred.colwise() - x);
  MatrixXd zdiff = (Zsig.colwise() - z_pred);

#ifdef NORMALIZE_ANGLE
  // angle normalization
  xdiff.row(3).unaryExpr([](double x) { return normalize(x); });

  if (n_z_ == 3) {
    zdiff.row(1).unaryExpr([](double x) { return normalize(x); });
  }
  if (n_z_ >= 5) {
    zdiff.row(3).unaryExpr([](double x) { return normalize(x); });
  }

#endif

  Tc = (xdiff.array().rowwise() * weights_.array().transpose()).matrix() *
       zdiff.transpose();

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

#ifdef NORMALIZE_ANGLE
  // angle normalization
  if (n_z_ == 3) {
    z_diff(1) = normalize(z_diff(1));
  }

  if (n_z_ >= 5) {
    z_diff(3) = normalize(z_diff(3));
  }

#endif

  // update state mean and covariance matrix
  x = x + K * z_diff;

#ifdef NORMALIZE_ANGLE
  x(3) = normalize(x(3));
#endif
  P = P - K * S * K.transpose();

  // print result
#ifdef DEBUG
  cout << "Updated state x: " << endl << x << endl;
  cout << "Updated state covariance P: " << endl << P << endl;
#endif

  // write result
  *x_out = x;
  *P_out = P;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and
   * radar measurements.
   */
  if (!is_initialized_) {
    VectorXd p(n_x_), q(n_x_);
    //
    p << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_,
        std_radrd_ * std_radrd_, std_radphi_ * std_radphi_,
        std_yawdd_ * std_yawdd_;

    q << 10, 10, 1000, 10, 10;

    P_ = (p.array() * q.array()).matrix().asDiagonal();

    x_ << 0, 0, 3, 0, 0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);

      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = rho_dot;
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);

      x_(0) = px;
      x_(1) = py;
    }

    z_.fill(0);

    if (n_z_ > 5) {
      meas_t_(0) = x_(0);
      meas_t_(1) = x_(1);
      meas_t_(2) = meas_package.timestamp_;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6; // sec
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  delta_t_ = delta_t;
  AugmentedSigmaPoints(&Xsig_aug_);

  SigmaPointPrediction(&Xsig_pred_);

  PredictMeanAndCovariance(&x_, &P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
#ifdef SENSOR_MIX

  double px = meas_package.raw_measurements_(0);
  double py = meas_package.raw_measurements_(1);

  // compute estimated velocity from history

  z_(0) = px;
  z_(1) = py;
  z_(2) = sqrt(px * px + py * py);
  z_(3) = atan2(py, px);

  if ((meas_package.timestamp_ - meas_t_(2)) > UPDATE_TIME_INTERVAL &&
      n_z_ > 5) {

    double dx = z_(0) - meas_t_(0);
    double dy = z_(1) - meas_t_(1);
    double t = (meas_package.timestamp_ - meas_t_(2)) / 1e6;

    z_(5) = sqrt(dx * dx + dy * dy) / t;

    // save into measurement history
    meas_t_(0) = z_(0);
    meas_t_(1) = z_(1);
    meas_t_(2) = meas_package.timestamp_;
    cout << "Velocity Meaurement: " << z_(5) << endl;
  }

#else
  n_z_ = 2;
  z_ = meas_package.raw_measurements_;
#endif
  // update measurements

  PredictMeasurement(&z_pred_, &S_, &Zsig_);

  UpdateState(&x_, &P_);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

#ifdef SENSOR_MIX
  // extract readings  for better readability
  double rho = meas_package.raw_measurements_(0);
  double phi = meas_package.raw_measurements_(1);
  double rho_dot = meas_package.raw_measurements_(2);

  z_(0) = rho * cos(phi);
  z_(1) = rho * sin(phi);
  z_(2) = rho;
  z_(3) = phi;
  z_(4) = rho_dot;

  if ((meas_package.timestamp_ - meas_t_(2)) > UPDATE_TIME_INTERVAL &&
      n_z_ > 5) {
    double dx = z_(0) - meas_t_(0);
    double dy = z_(1) - meas_t_(1);
    double dr = sqrt(dx * dx + dy * dy);

    double dt = (meas_package.timestamp_ - meas_t_(2)) / 1e6;

    z_(5) = dr / dt;

    meas_t_(0) = z_(0);
    meas_t_(1) = z_(1);
    meas_t_(2) = meas_package.timestamp_;
  }

#else

  n_z_ = 3;
  z_ = meas_package.raw_measurements_;

#endif

  PredictMeasurement(&z_pred_, &S_, &Zsig_);

  UpdateState(&x_, &P_);
}
