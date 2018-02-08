
#takes transcript as an input and gives out temporal features as output. Each temporal feature has its own individual funtion.
#transcript_file is of the format
#connection_uid,start_time,end_time,channel,utterance

import sys
import csv

def UsrTurnDur(transcript_file):
#Calculates duration of the customer turn in the call
	
	usr_turn_dur = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
                        speaker = line[3]
			if speaker == '2':
				usr_turn_dur = usr_turn_dur + (float(line[2])-float(line[1]))
	return usr_turn_dur

def SysTurnDur(transcript_file):
#Calculates duration of the agent turn in the call

	sys_turn_dur = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
                        speaker = line[3]
                        if speaker == '1':
                                sys_turn_dur = sys_turn_dur + (float(line[2])-float(line[1]))
        return sys_turn_dur

def StartTimeEndTime(transcript_file):
#returns start time and end time of the call

	countUtt = 0
	start_time = 0
	end_time = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
			utterance = line[4]
                        if utterance:
                                countUtt = countUtt + 1
                        if countUtt == 1:
                                start_time = float(line[1])
		end_time = float(line[2])
	return start_time, end_time
	
def NumOfSysWrd(transcript_file):
# Number of words spoken by the agent in the call
	
	NumWords = 0
	num_of_sys_wrds = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
			speaker = line[3]
			utterance = line[4]
			if speaker == '1':
				NumWords = len(utterance.split())
                                num_of_sys_wrds = num_of_sys_wrds + NumWords
	return num_of_sys_wrds
		
def NumOfSysTurn(transcript_file):
# Number of turns taken by the agent in the call; this is different that number of utterances

	lastspkr = '0'
	num_of_sys_turn = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
			speaker = line[3]
                        currentspkr = speaker
			if speaker == '1':
                                if currentspkr != lastspkr:
                                        num_of_sys_turn = num_of_sys_turn + 1
			lastspkr = currentspkr
	return num_of_sys_turn
	
def NumOfUsrWrds(transcript_file):
# Number of words spoken by the customer in the call

	NumWords = 0
	num_of_usr_wrds = 0
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
                        speaker = line[3]
                        utterance = line[4]
                        if speaker == '2':
                                NumWords = len(utterance.split())
                                num_of_usr_wrds = num_of_usr_wrds + NumWords
        return num_of_usr_wrds

def NumOfUsrTurn(transcript_file):
# Number of turns taken by the customer in the call; this is different that number of utterances

	lastspkr = '0'
	num_of_usr_turn = 0
        with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
                        speaker = line[3]
                        currentspkr = speaker
                        if speaker == '2':
                                if currentspkr != lastspkr:
                                        num_of_usr_turn = num_of_usr_turn + 1
                	lastspkr = currentspkr
        return num_of_usr_turn

def NumOverlaps(transcript_file):
	num_overlaps = 0
	previous_end_time = 0
	current_start_time = 0
	lastspkr = '0'
	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
			current_start_time = float(line[1])
			speaker = line[3]
                        currentspkr = speaker
			if currentspkr != lastspkr:
				if current_start_time < previous_end_time:
					num_overlaps = num_overlaps + 1
			previous_end_time = float(line[2])
			lastspkr = currentspkr
	return num_overlaps

def TotalTalkingTime(transcript_file):
# Calculates the total talking time of the agent and customer combined

	usr_turn_dur = UsrTurnDur(transcript_file)
	sys_turn_dur = SysTurnDur(transcript_file)
	total_talking_time = usr_turn_dur + sys_turn_dur
	return total_talking_time

def TimeOnTask(transcript_file):
# Calculates the total time of the call

	start_time,end_time = StartTimeEndTime(transcript_file)
	time_on_task = end_time - start_time
	return time_on_task

def MeanWrdsPerSysTurn(transcript_file):
# Calculates the mean number of words spoken by the agent per turn ( not utterance )

	num_of_sys_wrds = NumOfSysWrd(transcript_file)
	num_of_sys_turn = NumOfSysTurn(transcript_file)
	mean_wrds_per_sys_turn = float(num_of_sys_wrds)/num_of_sys_turn
	return mean_wrds_per_sys_turn

def MeanWrdsPerUsrTurn(transcript_file):
# Calculates the mean number of words spoken by the customer per turn ( not utterance )

	num_of_usr_wrds = NumOfUsrWrds(transcript_file)
        num_of_usr_turn = NumOfUsrTurn(transcript_file)
        mean_wrds_per_usr_turn = float(num_of_usr_wrds)/num_of_usr_turn
        return mean_wrds_per_usr_turn

def MeanSysTurnDur(transcript_file):
# Calculates the mean talking time of the agent per turn 

	sys_turn_dur = SysTurnDur(transcript_file)
	num_of_sys_turn = NumOfSysTurn(transcript_file)
	mean_sys_turn_dur = float(sys_turn_dur)/ num_of_sys_turn
	return mean_sys_turn_dur

def UsrRate(transcript_file):
# calculates average number of words per customer turn

	num_of_usr_wrds = NumOfUsrWrds(transcript_file)
	usr_turn_dur = UsrTurnDur(transcript_file)
	usr_rate = float(num_of_usr_wrds)/usr_turn_dur
	return usr_rate

def SysRate(transcript_file):
# calculates average number of words per agent turn

	num_of_sys_wrds = NumOfSysWrd(transcript_file)
	sys_turn_dur = SysTurnDur(transcript_file)
	sys_rate = float(num_of_sys_wrds)/sys_turn_dur
	return sys_rate

def UsrCallDom(transcript_file):
# Calculates the ratio of customer talking time to the total time of the call

	usr_turn_dur = UsrTurnDur(transcript_file)
	total_talking_time = TotalTalkingTime(transcript_file)
	usr_call_dominance = float(usr_turn_dur)/total_talking_time
	return usr_call_dominance

def SysCallDom(transcript_file):
# Calculates the ratio of agent talking time to the total time of the call
	
	sys_turn_dur = SysTurnDur(transcript_file)
	total_talking_time = TotalTalkingTime(transcript_file)
	sys_call_dominance = float(sys_turn_dur)/total_talking_time
	return sys_call_dominance

def AllTemporalFeatures(transcript_file):
# returns a list of all the temporal features
# returns a list of the format: [TimeOnTask,MeanWrdsPerSysTurn,MeanWrdsPerUsrTurn,MeanSysTurnDur,NumOverlaps,UsrCallDom,SysCallDom,UsrRate,SysRate]
	
	count_utt = 0
        previous_end_time = 0
        current_start_time = 0
        num_overlaps = 0
        num_of_sys_wrds = 0
        num_of_sys_utt = 0
        num_of_usr_wrds = 0
        num_of_usr_utt = 0
        sys_turn_dur = 0
        usr_turn_dur = 0
        num_of_sys_turn = 0
        num_of_usr_turn = 0
        lastspkr = 0
        currentspkr = 0
	num_words = 0
	total_talking_time = 0

	with open(transcript_file,'r') as test_file:
                test_csv = csv.reader(test_file, delimiter=',')
                for line_number,line in enumerate(test_csv):
                        speaker = line[3]
                        currentspkr = speaker
                        current_start_time = float(line[1])
                        utterance = line[4]
                        if utterance:
                                count_utt = count_utt + 1
                        if count_utt == 1:
                                start_time = float(line[1])
                        if speaker == '1':
                                if currentspkr != lastspkr:
                                        num_of_sys_turn = num_of_sys_turn + 1
                                        if current_start_time < previous_end_time:
                                                num_overlaps = num_overlaps + 1
                                sys_turn_dur = sys_turn_dur + (float(line[2])-float(line[1]))
                                num_of_sys_utt = num_of_sys_utt + 1
                                num_words = len(utterance.split())
                                num_of_sys_wrds = num_of_sys_wrds + num_words

                        if speaker == '2':
                                if currentspkr != lastspkr:
                                        num_of_usr_turn = num_of_usr_turn + 1
                                        if current_start_time < previous_end_time:
                                                num_overlaps = num_overlaps + 1
                                usr_turn_dur = usr_turn_dur + (float(line[2])-float(line[1]))
                                num_of_usr_utt = num_of_usr_utt + 1
                                num_words = len(utterance.split())
                                num_of_usr_wrds = num_of_usr_wrds + num_words
                        previous_end_time = float(line[2])
                        lastspkr = currentspkr
                end_time = float(line[2])

	total_talking_time = usr_turn_dur + sys_turn_dur
	time_on_task = end_time - start_time
	mean_wrds_per_sys_turn = float(num_of_sys_wrds)/num_of_sys_turn
	mean_wrds_per_usr_turn = float(num_of_usr_wrds)/num_of_usr_turn
	mean_sys_turn_dur = float(sys_turn_dur)/num_of_sys_turn
	usr_rate = float(num_of_usr_wrds)/usr_turn_dur
	sys_rate = float(num_of_sys_wrds)/sys_turn_dur
	usr_call_dom = float(usr_turn_dur)/total_talking_time
	sys_call_dom = float(sys_turn_dur)/total_talking_time
	return [time_on_task,mean_wrds_per_sys_turn,mean_wrds_per_usr_turn,mean_sys_turn_dur,num_overlaps,usr_call_dom,sys_call_dom,usr_rate,sys_rate]


#Use the below code to test the script
#transcript_file = sys.argv[1]
#total_talking_time = TotalTalkingTime(transcript_file)
#num_overlaps = NumOverlaps(transcript_file)
#time_on_task = TimeOnTask(transcript_file)
#mean_wrds_per_sys_turn = MeanWrdsPerSysTurn(transcript_file)
#mean_wrds_per_usr_turn = MeanWrdsPerUsrTurn(transcript_file)
#mean_sys_turn_dur = MeanSysTurnDur(transcript_file)
#usr_rate = UsrRate(transcript_file)
#sys_rate = SysRate(transcript_file)
#usr_call_dominance = UsrCallDom(transcript_file)
#sys_call_dom = SysCallDom(transcript_file)
#all_features = AllTemporalFeatures(transcript_file)
#print all_features


#print "total_talking_time",total_talking_time
#print "num_overlaps", num_overlaps
#print "time_on_task", time_on_task
#print "mean_wrds_per_sys_turn", mean_wrds_per_sys_turn
#print "mean_wrds_per_usr_turn",mean_wrds_per_usr_turn
#print "mean_sys_turn_dur",mean_sys_turn_dur
#print "usr_rate", usr_rate
#print "sys_rate", sys_rate
#print "usr_call_dominance", usr_call_dominance
#print "sys_call_dom", sys_call_dom

