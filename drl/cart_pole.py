import gym
from time import sleep
from pynput.keyboard import Listener

# direction
direc = 1
# termination flag 
terminate = True 

def main():
    global terminate

    def on_press(key, mod):
        global direc, terminate
        if key==97: direc = 0
        if key==100:  direc = 1
        if key==113: terminate = False
        print(f'Pressed {key} mod {mod}')
        
    # make environment of cart-pole, 
    env = gym.make('CartPole-v0')
    # termination condition of simulation for pole angle and pole velocity
    isterminate = lambda car_x, pole_angle, pole_velo: abs(car_x) <= 3 and abs(pole_angle) <= 30 and abs(pole_velo) <= 5
    
    env.reset()
    while terminate:
        env.render()
        env.unwrapped.viewer.window.on_key_press = on_press
        observ, reward, done, info = env.step(direc)
            
        sleep(0.1)

        car_x, pole_angle, pole_veloc = observ[0], observ[1], observ[2]
        terminate = isterminate(car_x, pole_angle, pole_veloc)

        print(f'car_x: {observ[0]:.3f}\tcar_v: {observ[1]:.3f}\tpole_ang: \
                {observ[2]:.3f}\tpole_v: {observ[3]:.3f}\treward: {reward}')
    print('Terminate')
    env.close()

if __name__ == "__main__":
    main()