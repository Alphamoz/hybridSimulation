class PID:
    # variables for velocity PID
    def __init__(self, maxVeloDes=0.1, arctan=20):
        self.maxVeloDes = maxVeloDes
        self.minVeloDes = 0
        self.minError = 0.2
        self.maxError = (arctan*maxVeloDes)-self.minVeloDes
        # self.minError = 0.2
        self.minmaxErrorDiff = self.maxError-self.minError
        self.errorDepth = 0

        # self.Kp = 100
        # self.Ki = 0
        # self.Kd = 2000

        self.last_error = 0
        self.integral = 0

    def calculatePID(self, setPoint, process_variable, Kp, Ki, Kd, lb, ub, limitlowval=True):
        t = 1
        error = setPoint - process_variable
        # Proportional term
        proportional = Kp * error
        # Integral term
        self.integral += error * t
        integral = Ki * self.integral
        # Derivative term
        derivative = Kd * (error - self.last_error) / t
        self.last_error = error
        # Calculate the control signal
        control_signal = proportional + integral + derivative
        if (control_signal > ub):
            control_signal = ub

        if (control_signal < lb):
            control_signal = lb

        if (limitlowval):
            if (control_signal > -5 and control_signal < 5):
                control_signal = 0

        return control_signal

    def calculate_desVelo(self, errorDepth):
        if (abs(errorDepth) > self.maxError):
            veloDes = self.maxVeloDes * (errorDepth)/abs(errorDepth)
            print("Velo menuju {}".format(veloDes))
        elif (abs(errorDepth) < self.minError):
            veloDes = 0
            print("Menuju nol velonya")
        else:
            veloDes = self.maxVeloDes/self.minmaxErrorDiff * \
                (errorDepth)-((self.minError*self.maxVeloDes/self.minmaxErrorDiff)
                              * (errorDepth)/abs(errorDepth))
            print("Transient velo")
        print(errorDepth, veloDes)
        return veloDes
