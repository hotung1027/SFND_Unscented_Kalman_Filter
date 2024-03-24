#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
class UKF {
public:
  /**
   * Constructor
   */
  UKF();
  /**
   * Destructor
   */
  virtual ~UKF();
  /**
   * Init
   * @param    MatrixXd    The initial state covariance matrix*/

  void AugmentedSigmaPoints(MatrixXd *Xsig_out);

  void SigmaPointPrediction(MatrixXd *Xsig_out);
  void PredictMeanAndCovariance(VectorXd *x_out, MatrixXd *P_out);

  void PredictMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd *Zsig_out);

  void UpdateState(VectorXd *x_out, MatrixXd *P_out);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or
   * laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser
   * measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar
   * measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  Eigen::VectorXd z_;
  Eigen::VectorXd z_pred_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // augmented mean predicted sigma points matrix
  Eigen::MatrixXd Xsig_aug_;

  // sigma point prediction matrix for measurements
  Eigen::MatrixXd Zsig_;

  // measurement covariance matrix S
  Eigen::MatrixXd S_;

  // measurement noise covariance matrix
  Eigen::MatrixXd R_;
  // measurement noise covariance matrix for radar
  Eigen::MatrixXd R_radar_;
  // measurement noise covariance matrix for lidar
  Eigen::MatrixXd R_lidar_;

  // time when the state is true, in us
  long long time_us_;

  // time difference between k and k+1 in s
  double delta_t_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // measurement dimension
  int n_z_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;
};

#endif // UKF_H
