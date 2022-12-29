import numpy as np
import pybullet as p
import os
import lib
from urdfenvs.robots import prius

class MovingObstacleTIntersection:
    def __init__(self, trajectory_type, dt, direction=None, speed=None, offset=None):
        self.start_position = [-direction * 10, -direction * 1.25, 0]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2 - direction * np.pi / 2])
        self.dt = dt
        self.direction = 1 if None else direction  # 1 for facing right, -1 for facing left
        assert np.abs(self.direction) == 1, "direction should either be 1 or -1"
        self.speed = 13.9 if None else speed  # 50 km/h
        self.counter = 0 if offset is None else - int(offset / dt)
        self.id = None
        self.steps = None
        self.traj_position = None
        self.traj_orientation = None
        self.done = False
        # self.urdf = "/home/christiaan/gym_env_urdf/gym_envs_urdf/urdfenvs/robots/prius/prius.urdf"
        self.urdf = os.path.join(os.path.dirname(prius.__file__), 'prius.urdf')
        # self.urdf = '/home/christiaan/Robotics/2022-2023/Q2/RO47005_Planning_and_Decision_Making/ro47005-project/ro47005-project/lib/prius.urdf'
        # Create trajectories
        self.trajectory(trajectory_type, self.direction, self.speed)

    def add_env(self):
        self.id = p.loadURDF(fileName=self.urdf,
                             basePosition=self.start_position,
                             baseOrientation=self.start_orientation,
                             globalScaling=0.3,
                             )

    def trajectory(self, trajectory_type, direction, speed):
        if trajectory_type == "straight":
            self.trajectory_straight(direction, speed)
        elif trajectory_type == "turn":
            self.trajectory_turn(direction, speed)
        else:
            raise ValueError("No valid trajectory type chosen!")

    def trajectory_straight(self, direction, speed):
        """" Returns an array with all the points on the trajectory. """
        distance = 20
        self.steps = int(distance / (speed * self.dt))
        self.traj_position = np.vstack(
            (direction * np.linspace(0, distance, self.steps), np.zeros(self.steps),
             np.zeros(self.steps))).T + self.start_position
        self.traj_orientation = np.repeat(np.expand_dims(self.start_orientation, axis=0), self.steps, axis=0)

    def trajectory_turn(self, direction, speed):
        """ Returns an array with all the points on the trajectory. """
        if direction == 1:
            # part 1
            distance1 = 7
            steps1 = int(distance1 / (speed * self.dt))
            positions1 = np.vstack(
                (direction * np.linspace(0, distance1, steps1), np.zeros(steps1),
                 np.zeros(steps1))).T + self.start_position
            orientations1 = np.repeat(np.expand_dims(self.start_orientation, axis=0), steps1, axis=0)
            # part 2 (corner)
            corner_radius = 1.75
            distance2 = corner_radius * np.pi / 2
            steps2 = int(distance2 / (speed * self.dt))
            angles = np.linspace(0, np.pi / 2, steps2)
            positions2 = np.vstack((corner_radius * np.sin(angles), corner_radius * (np.cos(angles) - 1), np.zeros(steps2))).T + positions1[-1]
            orientations2 = np.array([p.getQuaternionFromEuler([0, 0, -a]) for a in angles])
            # part 3
            distance3 = 7
            steps3 = int(distance3 / (speed * self.dt))
            positions3 = np.vstack((np.zeros(steps3), -np.linspace(0, distance3, steps3), np.zeros(steps3))).T + \
                         positions2[-1]
            orientations3 = np.repeat(np.expand_dims(orientations2[-1], axis=0), steps3, axis=0)

            self.steps = steps1 + steps2 + steps3
            self.traj_position = np.vstack((positions1, positions2, positions3))
            self.traj_orientation = np.vstack((orientations1, orientations2, orientations3))
        elif direction == -1:
            # part 1
            distance1 = 7
            steps1 = int(distance1 / (speed * self.dt))
            positions1 = np.vstack(
                (direction * np.linspace(0, distance1, steps1), np.zeros(steps1),
                 np.zeros(steps1))).T + self.start_position
            orientations1 = np.repeat(np.expand_dims(self.start_orientation, axis=0), steps1, axis=0)
            # part 2 (corner)
            corner_radius = 4.25
            distance2 = corner_radius * np.pi / 2
            steps2 = int(distance2 / (speed * self.dt))
            angles = np.linspace(0, np.pi / 2, steps2)
            positions2 = np.vstack((-corner_radius * np.sin(angles), corner_radius * (np.cos(angles) - 1), np.zeros(steps2))).T + positions1[-1]
            orientations2 = np.array([p.getQuaternionFromEuler([0, 0, a - np.pi]) for a in angles])
            # part 3
            distance3 = 7
            steps3 = int(distance3 / (speed * self.dt))
            positions3 = np.vstack((np.zeros(steps3), -np.linspace(0, distance3, steps3), np.zeros(steps3))).T + \
                         positions2[-1]
            orientations3 = np.repeat(np.expand_dims(orientations2[-1], axis=0), steps3, axis=0)

            self.steps = steps1 + steps2 + steps3
            self.traj_position = np.vstack((positions1, positions2, positions3))
            self.traj_orientation = np.vstack((orientations1, orientations2, orientations3))
        else:
            raise NotImplementedError("Direction should either be 1 or -1 when trajectory_type=\'turn\'")

    def step(self):
        """ Changes the position of the moving obstacle. Needs to be called in every timestep. """
        if self.counter >= len(self.traj_position):
            if self.done is not True:
                p.removeBody(self.id)
                self.done = True
        elif self.counter < 0:
            self.counter += 1
        elif self.counter == 0:
            self.add_env()
            self.counter += 1
        else:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.id,
                                              posObj=self.traj_position[self.counter],
                                              ornObj=self.traj_orientation[self.counter],
                                              )
            self.counter += 1
