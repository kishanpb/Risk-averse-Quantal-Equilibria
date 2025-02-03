import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ArrowStyle
from matplotlib.path import Path
import seaborn as sns

colors = sns.color_palette("deep", 10)
colors2=sns.color_palette("RdGy", 10)[6]
colors3=sns.color_palette("RdGy", 10)[9]
colors4=sns.color_palette("RdGy", 10)[4]
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

delta=1e-15
# Testworld=(np.zeros((5,4)),['0___','#___','#__#','#___','1___'])


#Small
#Testworld=(np.zeros((3,3)),['1__','#__','0__'])
#Testworld=(np.zeros((3,3)),['___','#__','$__'])

#Med
Testworld=(np.zeros((4,3)),['1__','#__','#__','0__'])
#Testworld=(np.zeros((3,3)),['___','#__','$__'])


def proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = np.reshape(v, (v.shape[0]))
    n,   = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.all(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def gradient_player(M,p,x,eps):
	return -M.dot(p)-eps*1/(x+delta)

def gradient_adv(M,p,x,y,tau):

	return M.T.dot(x)+1/tau*(np.log((p+delta)/(y+delta))+1)



def solve_game(M1,M2,tau,eps,tau2,eps2,T,gamma=0.1,show=False,tol=1e-4):

	(n,m)=M1.shape

	x=proj_simplex(np.ones([n,1]))
	px=proj_simplex(np.ones([m,1]))
	y=proj_simplex(np.ones([m,1]))
	py=proj_simplex(np.ones([n,1]))

	payoffs1=[]
	payoffs2=[]
	strats1=np.zeros((n,T))
	strats2=np.zeros((m,T))
	lastx=0
	lasty=0
	lasti=100
	for t in range(T):
		x=proj_simplex(x-gamma*gradient_player(M1,px,x,eps))
		px=proj_simplex(px-gamma*gradient_adv(M1,px,x,y,tau))
		y=proj_simplex(y-gamma*gradient_player(M2,py,y,eps2))
		py=proj_simplex(py-gamma*gradient_adv(M2,py,y,x,tau2))

		payoffs1.append(-x.T.dot(M1).dot(px)-1/tau*(np.sum(y*np.log((y+delta)/(px+delta)))))
		payoffs2.append(-y.T.dot(M2).dot(py)-1/tau2*(np.sum(x*np.log((x+delta)/(py+delta)))))
		strats1[:,t]=x[:]
		strats2[:,t]=y[:]
		diff1=np.abs(x-lastx)
		diff2=np.abs(y-lasty)
		lastx=x
		lasty=y
		if np.sum(diff1)<tol and np.sum(diff2)<tol:
			if t>=101:
				lasti=100
			else:
				lasti=t
			break

	payoff1=np.mean(payoffs1[-lasti:])
	payoff2=np.mean(payoffs2[-lasti:])
	strat1=np.mean(strats1,axis=1)
	strat2=np.mean(strats2,axis=1)
	if show:
		plt.plot(payoffs1)
		plt.pause(0.1)
	return payoff1,payoff2, x,y




class MAGridWorld:
	
	def __init__(self,world,horizon,taus,eps):

		y,x=world[0].shape
		self.tau1,self.tau2=taus
		self.eps1,self.eps2=eps
		self.x=x
		self.y=y
		self.num_actions=4
		self.desc=world[1]
		self.num_agents=2
		self.horizon=horizon
		self.gamma=1.0

		if len(self.desc)!=self.y:
			raise Exception('Error: World Description and Grid do not match') 
		
		self.final_bad_states=[]
		self.final_good_states=[[] for a in range(self.num_agents)]

		self.final_bad_nums=[]
		self.final_good_nums=[[] for a in range(self.num_agents)]

		for j in range(self.y):
			if len(self.desc[j])!=self.x:
				raise Exception('Error: World Description and Grid do not match') 
			else:
				for i in range(self.x):
					if self.desc[j][i]=='#':
						self.final_bad_states.append((i,self.y-j-1))
					for a in range(self.num_agents):
						if self.desc[j][i]==str(a):
							self.final_good_states[a].append((i,self.y-j-1))
					if self.desc[j][i]=='$':
						for a in range(self.num_agents):
							self.final_good_states[a].append((i,self.y-j-1))


		self.state2num={}
		self.num2state={}
		count=0
		for j in range(self.y):
			for i in range(self.x):
				self.state2num[(i,j)]=count
				self.num2state[count]=(i,j)
				if (i,j) in self.final_bad_states:
					self.final_bad_nums.append(count)
				for a in range(self.num_agents):
					if (i,j) in self.final_good_states[a]:
						self.final_good_nums[a].append(count)
				count+=1
		
		self.num_states=count

		self.actions={0:(1,0),1:(0,1),2:(-1,0),3:(0,-1)}


		self.Ps={}
		for s1 in range(self.num_states):
			for s2 in range(self.num_states):
				for a1 in range(self.num_actions):
					for a2 in range(self.num_actions):
						self.Ps[(s1,s2,a1,a2)]=self.getPs(s1,s2,a1,a2)
		
	

	def getPs(self,s1,s2,a1,a2):

		ps=np.zeros((self.num_states,self.num_states))
		state1=self.num2state[s1]
		state2=self.num2state[s2]
		true_prob=0.9
		if np.abs(state1[0]-state2[0])<=1 and np.abs(state1[1]-state2[1])<=1:
			true_prob=0.5

		goals=[s1,s2]

		possible1=[]
		possible2=[]
		if s1 not in self.final_bad_nums and s1 not in self.final_good_nums[0]:
			for action in self.actions.keys():
				d1,d2=self.actions[action]
				next_s1=(state1[0]+d1,state1[1]+d2)
				if next_s1 in self.state2num.keys():
					ns1=self.state2num[next_s1]
					possible1.append(ns1)
					if a1==action:
						goals[0]=ns1
		else:
			possible1=[s1]

		if s2 not in self.final_bad_nums and s2 not in self.final_good_nums[1]:
			for act2 in self.actions.keys():
				d11,d22=self.actions[act2]
				next_s2=(state2[0]+d11,state2[1]+d22)
				if next_s2 in self.state2num.keys():
					ns2=self.state2num[next_s2]
					possible2.append(ns2)
					if act2==a2:
						goals[1]=ns2
		else:
			possible2=[s2]

		for i in possible1:
			for j in possible2:
				ps[i,j]=(1-true_prob)/(len(possible1)*len(possible2))

		ps[goals[0],goals[1]]+=true_prob
		return ps

	def getRs(self,s1,s2,a1,a2):

		Rs1=0.1
		Rs2=0.1

		if s1 in self.final_good_nums[0]:
			Rs1=1
		elif s1 in self.final_bad_nums:
			Rs1=-2
		if s2 in self.final_good_nums[1]:
			Rs2=1
		elif s2 in self.final_bad_nums:
			Rs2=-2

		return Rs1,Rs2


	def KL(self,Qs,Ps,tau):

		return -1/tau*np.log(np.sum(Ps*np.exp(tau*Qs)[:,:,0])+delta)


	def qlearn(self,T=200,gamma=0.0001,show=False,tol=1e-4):

		self.payoffs=[[np.zeros((self.num_states,self.num_states,1)) for t in range(self.horizon)] for a in range(self.num_agents)]
		
		self.Qs=[[np.zeros([self.num_states,self.num_states,self.num_actions,self.num_actions]) for t in range(self.horizon)] for ag in range(self.num_agents)]

		self.policies=[[np.zeros((self.num_states,self.num_states,self.num_actions)) for t in range(self.horizon)] for ag in range(self.num_agents)]

		print(0)
		for s1 in range(self.num_states):
				for s2 in range(self.num_states):
					for a1 in range(self.num_actions):
						for a2 in range(self.num_actions):
							R1,R2=self.getRs(s1,s2,a1,a2)
							self.Qs[0][self.horizon-1][s1,s2,a1,a2]=R1
							self.Qs[1][self.horizon-1][s1,s2,a1,a2]=R2

		for t in range(self.horizon-1):
			print(t+1)
			for s1 in range(self.num_states):
				for s2 in range(self.num_states):
					p1,p2,pi1,pi2=solve_game(self.Qs[0][self.horizon-t-1][s1,s2,:,:],self.Qs[1][self.horizon-t-1][s1,s2,:,:].T,self.tau1,self.eps1,self.tau2,self.eps2,T,gamma/(t+1),show,tol)

					self.payoffs[0][self.horizon-t-1][s1,s2]=p1
					self.payoffs[1][self.horizon-t-1][s1,s2]=p2
					self.policies[0][self.horizon-t-1][s1,s2,:]=pi1
					self.policies[1][self.horizon-t-1][s1,s2,:]=pi2
			
			for s1 in range(self.num_states):
				for s2 in range(self.num_states):
					for a1 in range(self.num_actions):
						for a2 in range(self.num_actions):
							R1,R2=self.getRs(s1,s2,a1,a2)
							self.Qs[0][self.horizon-t-2][s1,s2,a1,a2]=R1+self.gamma*self.KL(self.payoffs[0][self.horizon-t-1],self.Ps[(s1,s2,a1,a2)],self.tau1)
							self.Qs[1][self.horizon-t-2][s1,s2,a1,a2]=R2+self.gamma*self.KL(self.payoffs[1][self.horizon-t-1],self.Ps[(s1,s2,a1,a2)],self.tau2)
		print(t+2)
		for s1 in range(self.num_states):
			for s2 in range(self.num_states):
				p1,p2,pi1,pi2=solve_game(self.Qs[0][0][s1,s2,:,:],self.Qs[1][0][s1,s2,:,:].T,self.tau1,self.eps1,self.tau2,self.eps2,T,gamma/self.horizon,show,tol)

				self.payoffs[0][0][s1,s2]=p1
				self.payoffs[1][0][s1,s2]=p2
				self.policies[0][0][s1,s2,:]=pi1
				self.policies[1][0][s1,s2,:]=pi2
		return

	def viewWorld(self,initial=None):

		self.fig=plt.figure()
		self.gca=self.fig.gca()
		plt.gca().set_aspect('equal', adjustable='box')
		plt.xlim([0,self.x])
		plt.ylim([0,self.y])
		self.gca.set_xticks([])
		self.gca.set_yticks([])
		pointer=[0,0]
		
		for i in range(self.x):
			for j in range(self.y):
				center=(i,j) 
				if (i,j) in self.final_bad_states:
					self.gca.add_patch(Rectangle(center, 1, 1, facecolor=colors3,ec='k'))
				elif (i,j) in self.final_good_states[0] and (i,j) in self.final_good_states[1]:
					self.gca.add_patch(Rectangle(center, 1, 1, facecolor=colors[1],ec='k'))
				elif (i,j) in self.final_good_states[0] and  not (i,j) in self.final_good_states[1]:
					self.gca.add_patch(Rectangle(center, 1, 1, facecolor='r',ec='k'))
				elif (i,j) not in self.final_good_states[0] and  (i,j) in self.final_good_states[1]:
					self.gca.add_patch(Rectangle(center, 1, 1, facecolor='b',ec='k'))
				else:
					self.gca.add_patch(Rectangle(center, 1, 1, facecolor=colors2,ec='k'))


				if initial is not None:
					if (i,j) in self.initial[0]:
						self.gca.add_patch(Rectangle(center, 1, 1, facecolor='r',ec='k'))
					elif (i,j) in self.initial[1]:
						self.gca.add_patch(Rectangle(center, 1, 1, facecolor='b',ec='k'))

				pointer[1]+=1;
			pointer[0]+=1
			pointer[1]=0

	def viewPath(self,initial):

		self.viewWorld()
		state1=initial[0]
		state2=initial[1]
		t=0
		vertices=[]
		code=[Path.MOVETO]
		update1=True
		update2=True

		for t in range(self.horizon):
			start1=(state1[0]+0.5,state1[1]+0.5)
			start2=(state2[0]+0.5,state2[1]+0.5)

			action1=np.argmax(self.policies[0][t][self.state2num[state1],self.state2num[state2],:])
			action2=np.argmax(self.policies[1][t][self.state2num[state1],self.state2num[state2],:])

			nextstate1=(state1[0]+self.actions[action1][0],state1[1]+self.actions[action1][1])
			nextstate2=(state2[0]+self.actions[action2][0],state2[1]+self.actions[action2][1])

			if nextstate1 in self.state2num.keys() and state1 not in self.final_bad_states and state1 not in self.final_good_states[0] :
				end1=(start1[0]+self.actions[action1][0],start1[1]+self.actions[action1][1])
				state1=nextstate1
				arrow1=FancyArrowPatch((start1[0],start1[1]),(end1[0],end1[1]),lw=2,facecolor='darkred',ec='darkred',arrowstyle=ArrowStyle('->,head_length=4,head_width=4'))
			else:
				end1=start1
				arrow1=Circle((start1[0],start1[1]),0.01,lw=1,facecolor='darkred',ec='darkred')

			if nextstate2 in self.state2num.keys() and state2 not in self.final_bad_states and state2 not in self.final_good_states[1]:
				end2=(start2[0]+self.actions[action2][0],start2[1]+self.actions[action2][1])
				state2=nextstate2
				arrow2=FancyArrowPatch((start2[0],start2[1]),(end2[0],end2[1]),lw=2,facecolor='darkblue',ec='darkblue',arrowstyle=ArrowStyle('->,head_length=4,head_width=4'))

			else:
				end2=start2
				arrow2=Circle((start2[0],start2[1]),0.01,lw=1,facecolor='darkblue',ec='darkblue')


			if update1:
				self.gca.add_patch(arrow1)
			if update2:
				self.gca.add_patch(arrow2)



### Plot Configuration
m1=MAGridWorld(Testworld,15,[0.1,0.1],[10,10])
m1.qlearn(1000,0.001,1,1e-5)



# m2=MAGridWorld(Testworld,15,[0.001,0.001],[10,10])
# m2.qlearn(1000,0.0005,1,1e-5)








