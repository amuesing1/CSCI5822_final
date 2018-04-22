import sys
sys.path.append("../../../scenario_simulator/modules/")
from DynamicsProfiles import *


def buildHistoricalModels(filePath):

    models = ['Cirrus', 'Stratus', 'Cumulus', 'Nimbostratus', 'Cumulonimbus', 'Altostratus']

    histModels = {}

    warnings.filterwarnings("ignore")
    with h5py.File(filePath, 'r') as infile:
      for mod in models:
        print("Now building model {} from historical data.".format(mod))
        allData = []
        allLengths = []
        l = len(infile[mod].values())
        #for c in infile[mod].values():
        h = infile[mod].values()
        for c in range(int(l/10)):
          allData.append(h[c][0:-1].tolist())
          allLengths.append(len(h[c][0:-1]))

        for i in range(len(allData)):
          for j in range(len(allData[i])):
            allData[i][j] = [allData[i][j]]

        allData = np.concatenate(allData)


        allModels = []
        allBIC = []
        for i in range(2,10):
          a = hmm.GaussianHMM(n_components=i).fit(allData,allLengths)
          allModels.append(a)
          llh,posteriors = a.score_samples(allData[0])
          bic = np.log(len(allData[0]))*i - 2*llh
          allBIC.append(bic)
        bestModel = allModels[np.argmin(allBIC)]

        #bestModel = hmm.GaussianHMM(n_components=5).fit(allData,allLengths)

        best = {}
        best['transition'] = bestModel.transmat_.tolist()
        best['prior'] = bestModel.startprob_.tolist()


        means = bestModel.means_.tolist()
        var = bestModel.covars_.tolist()
        obs = []
        for i in range(len(means)):
          obs.append(GM(means[i],var[i],1))

        best['obs'] = obs


        histModels[mod] = best

    saveFile = open('./histModels_normed.npy','w')
    np.save(saveFile,histModels)




def buildNewHistoricalModels(dataSet, saveFileName='../data/testing/histModels_fams_2.npy', sizes=300):
    models = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']

    histModels = {}
    warnings.filterwarnings("ignore")
    for mod in models:
        print("Now building model {} from historical data.".format(mod))
        allFamData = []
        allFamLengths = []
        for i in range(5):
            #print("Genus: {}".format(i))
            allTypeData = []
            allTypeLengths = []
            h = dataSet[mod][str(i)]
            for j in range(sizes):
                allFamData.append(h[j][0:-1].tolist())
                allFamLengths.append(len(h[j][0:-1]))
                allTypeData.append(h[j][0:-1].tolist())
                allTypeLengths.append(len(h[j][0:-1]))


            for k in range(len(allTypeData)):
                for j in range(len(allTypeData[k])):
                    allTypeData[k][j] = [allTypeData[k][j]]

            allTypeData = np.concatenate(allTypeData)

            allModels = []
            allBIC = []
            for j in range(2,9):
                a = hmm.GaussianHMM(n_components=j).fit(allTypeData,allTypeLengths)
                allModels.append(a)
                llh,posteriors = a.score_samples(allTypeData[0])
                bic = 0
                for k in range(len(allTypeData)):
                    bic += np.log(len(allTypeData[k]))*j - 2*llh
                bic = bic/len(allTypeData)
                allBIC.append(bic)

            bestModel = allModels[np.argmin(allBIC)]
            best = {}
            best['transition'] = bestModel.transmat_.tolist()
            best['prior'] = bestModel.startprob_.tolist()

            means = bestModel.means_.tolist()
            var = bestModel.covars_.tolist()
            obs = []
            for j in range(len(means)):
                obs.append(GM(means[j],var[j],1))

            best['obs'] = obs

            nam = mod + str(i)
            print("Completed Profile: {}".format(nam))
            histModels[nam] = best


        for k in range(len(allFamData)):
            for j in range(len(allFamData[k])):
                allFamData[k][j] = [allFamData[k][j]]
        allFamData = np.concatenate(allFamData)

        allModels = []
        allBIC = []
        for j in range(2,15):
            a = hmm.GaussianHMM(n_components=j).fit(allFamData,allFamLengths)
            allModels.append(a)
            llh,posteriors = a.score_samples(allFamData[0])
            bic = 0
            for k in range(len(allFamData)):
                bic += np.log(len(allFamData[k]))*j - 2*llh
            bic = bic/len(allFamData)
            allBIC.append(bic)
        bestModel = allModels[np.argmin(allBIC)]
        best = {}
        best['transition'] = bestModel.transmat_.tolist()
        best['prior'] = bestModel.startprob_.tolist()


        means = bestModel.means_.tolist()
        var = bestModel.covars_.tolist()
        obs = []
        for j in range(len(means)):
            obs.append(GM(means[j],var[j],1))

        best['obs'] = obs

        nam = mod
        print("Completed Profile: {}".format(nam))
        histModels[nam] = best

    saveFile = open('../data/testing/histModels_fams_2.npy','w')
    np.save(saveFile,histModels)

def buildDataSet(size=300, weather=True):

    #get an intensity from each
    models = [Stratiform, Cirriform, Stratocumuliform, Cumuliform, Cumulonibiform]
    #  models = [Cumulonibiform]
    subs = [str(i) for i in range(5)]

    allSeries = {}

    baseSeries = {}
    for mod in models:
        #baseSeries[mod.__name__] = {}
        allSeries[mod.__name__] = {}
        for i in range(5):
            a = mod(genus = i,weather=weather)
            allSeries[mod.__name__][str(i)] = []
            for j in range(size):
                b = deepcopy(a.intensityModel)
                c = b+np.random.normal(0,2,(len(b)))
                for k in range(len(c)):
                    c[k] = max(c[k],1)

                allSeries[mod.__name__][str(i)].append(c)

    return allSeries


def classifyTimeSeries(testDataIn,histModel):
    #run forward
    suma = 0
    results = {}
    for key in histModel.keys():
        results[key] = forward(testDataIn['intensities'],histModel[key])
        suma += results[key]
    for key in results.keys():
        results[key] = results[key]/suma
    return results



def testAllData(filePath,models):

    profiles = ['Cirrus', 'Stratus', 'Cumulus', 'Nimbostratus', 'Cumulonimbus', 'Altostratus']

    allProfs = {}

    with h5py.File(filePath, 'r') as infile:
        for prof in profiles:
            allData = []
            l = len(infile[prof].values())
            #for c in infile[prof].values():
            h = infile[prof].values()
            for c in range(int(l/10),l):
                allData.append(h[c][0:-1].tolist())
                #allData.append(a)

            allProfs[prof] = allData

    results = np.zeros(shape=(len(profiles),len(profiles)))

    for i in range(len(profiles)):
        print("Now Testing: {}".format(profiles[i]))
        #print(len(allProfs[profiles[i]]))
        #l = len(allProfs[profiles[i]])
        for j in range(len(allProfs[profiles[i]])):
        #for j in range(int(l*9/10),l):
            testIn = {'intensities':allProfs[profiles[i]][j]}
            tmp = classifyTimeSeries(testIn,models)
            #find max:
            best = max(tmp, key=tmp.get)
            #print(len(allProfs[profiles[i]]))
            results[i][profiles.index(best)] += 1/len(allProfs[profiles[i]])
            #for key in tmp.keys():
                #results[i][profiles.index(key)] += tmp[key]/len(allProfs[profiles[i]])

    print(results)

    plotProfileNames = ['','Cirrus', 'Stratus', 'Cumulus', 'Nimbostratus', 'Cumulonimbus', 'Altostratus']

    fig,ax = plt.subplots()
    ax.imshow(results)
    ax.set_xticklabels(plotProfileNames)
    ax.set_yticklabels(plotProfileNames)
    plt.suptitle('Confusion Matrix for HMM Classificaiton')
    ax.set_ylabel("True State")
    ax.set_xlabel("Predicted State")

    plt.show()



def loadHistoricalModels(fileName):
    return np.load(fileName).item()

def forward(dataIn,model):
    numSteps = len(dataIn)+1

    x0 = model['prior']
    pxx = model['transition']
    pyx = model['obs']

    numStates = len(x0)


    #forward step
    alphas = [[-1]*numStates]
    for i in range(1, numSteps):
        alphas.append([-1]*numStates)
    alphas[0] = x0
    for i in range(1, len(alphas)):
        for xcur in range(len(alphas[0])):
            alphas[i][xcur] = 0
            for xprev in range(len(alphas[0])):
                alphas[i][xcur] += alphas[i-1][xprev]*pxx[xcur][xprev]
            alphas[i][xcur] = alphas[i][xcur]*pyx[xcur].pointEval(dataIn[i-1])

    return(sum(alphas[-1]))

def continueForward(newData, model, prevAlpha=[-1,-1]):
    x0 = model['prior']
    pxx = model['transition']
    pyx = model['obs']

    numStates = len(x0)
    if prevAlpha[0] == -1:
        prevAlpha=x0

    newAlpha = [-1]*numStates
    for xcur in range(numStates):
        newAlpha[xcur] = 0
        for xprev in range(numStates):
            newAlpha[xcur] += prevAlpha[xprev]*pxx[xcur][xprev]
        newAlpha[xcur] = newAlpha[xcur]*pyx[xcur].pointEval(newData)
    return newAlpha

def newClassify(testDataIn,histModel,modNames):
    #run forward
    suma = 0
    results = {}
    for modName in modNames:
        results[modName] = forward(testDataIn,histModel[modName])
        suma += results[modName]
    for key in results.keys():
        results[key] = results[key]/suma
    return results

def newTestAllData(models,dataSet,sizes):
    famNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']
    #typeNames = [str(i) for i in range(5)]
    #for every dataset
    #test which family it comes from
    #test which type it is in that dataset
    #report two confusion matrices

    allFamProfs = {}
    famResults = np.zeros(shape=(len(famNames),len(famNames)))
    typeResults = np.zeros(shape=(len(famNames),5,5))
    # Jeremy's attempt
#      typeResults = np.zeros(shape=(5,5))
#      for i in range(2):
#          print("Genus: {}".format(i))
#          for j in range(sizes):
#              testIn = dataSet['Cumulonibiform'][str(i)][j]
#              typeNames = ['Cumulonibiform'+str(k) for k in range(5)]
#              tmp = newClassify(testIn,models,typeNames)
#              print tmp
#              best2 = max(tmp,key=tmp.get)
#              typeResults[i][typeNames.index(best2)] += 1/sizes
    for mod in famNames:
        print("Now testing: {}".format(mod))
        for i in range(1):
            print("Genus: {}".format(i))
            #nam = mod+str(i)

            for j in range(sizes):
                # print(mod)
                # print(str(i))
                testIn = dataSet[mod][str(i)][j]
                #find out which family it thinks it came from
                tmp = newClassify(testIn,models,famNames)
                best = max(tmp, key=tmp.get)
                famResults[famNames.index(mod)][famNames.index(best)] += 5/sizes


                #if it came from the right family, see if it's the right one
                if best == mod:
                    typeNames = [mod+str(k) for k in range(5)]
                    tmp = newClassify(testIn,models,typeNames)
                    best2 = max(tmp,key=tmp.get)
                    typeResults[famNames.index(mod)][i][typeNames.index(best2)] += 1/sizes


    #  normalize all the type results
    for i in range(len(famNames)):
        for j in range(5):
            suma = sum(typeResults[i][j])
            for k in range(5):
                typeResults[i][j][k] = typeResults[i][j][k]/suma
# more of Jeremy's attempt
#      for j in range(5):
#          suma = sum(typeResults[j])
#          for k in range(5):
#              typeResults[j][k] = typeResults[j][k]/suma

    #  normalize all fam results
    for i in range(len(famNames)):
        suma = sum(famResults[i])
        for j in range(len(famNames)):
            famResults[i][j] = famResults[i][j]/suma

    print("Families:")
    print(famResults)
    print("")

    plotFamNames = ['','Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']

    fig,ax = plt.subplots()
    ax.imshow(famResults)
    ax.set_xticklabels(plotFamNames)
    ax.set_yticklabels(plotFamNames)
    plt.suptitle('All Family Confusion Matrix for HMM Classificaiton')
    ax.set_ylabel("True State")
    ax.set_xlabel("Predicted State")

    plt.savefig('../img/familyConfusion_2.png')

    plotTypeNames = ['','0','1','2','3','4']
    for i in range(5):
        print("Types:{}".format(famNames[i]))
        print(typeResults[i])
        print("")
        fig,ax = plt.subplots()
        ax.imshow(typeResults[i])
        ax.set_xticklabels(plotTypeNames)
        ax.set_yticklabels(plotTypeNames)
        plt.suptitle('{} Family Confusion Matrix for HMM Classificaiton'.format(famNames[i]))
        ax.set_ylabel("True State")
        ax.set_xlabel("Predicted State")

        plt.savefig('../img/{}_confusion_2.png'.format(famNames[i]))

    saveFile = open('../data/testing_confuse_fam_2.npy','w')
    np.save(saveFile,famResults)

    saveFile = open('../data/testing_confuse_types_2.npy','w')
    np.save(saveFile,typeResults)

def makingAndTestingStuff():
    weather = True

    #  modelFileName = './histModels_fams_2.npy'
    #  sizeTrain = 100
    #  dataSetTrain = buildDataSet(sizeTrain,weather)
    #  buildNewHistoricalModels(dataSetTrain,modelFileName,sizeTrain)

    # beep = lambda x: os.system("echo -n '\a'sleep 0.6" * x)
    # beep(5)


    sizeTest = 100
    dataSetTest = buildDataSet(sizeTest,weather)
    models = loadHistoricalModels('../data/histModels_fams.npy')

    # modelNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']
    # for nam in modelNames:
    # 	print(len(models[nam]['obs']))

    newTestAllData(models,dataSetTest,sizeTest)

def humanTesting():
    modelFileName = '../data/histModels_fams.npy'
    models = loadHistoricalModels(modelFileName)

    genus = 'Cirriform0'
    species = Cirriform(genus = 0,weather=False)
    data = species.intensityModel

    famNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']

    obsMod = {}
    obsMod['Stratiform'] = [0.1442307692,0.0721153846,0.0120192308,0.0600961538,0.0480769231,
                          0.0240384615,0.0360576923,0.0120192308,0.1201923077,0.0600961538,
                          0.0360576923,0.1442307692,0.0024038462,0.0120192308,0.2163461538]
    obsMod['Cirriform'] = [0.0748129676,0.0997506234,0.0623441397,0.1496259352,0.0872817955,
                        0.0124688279,0.0249376559,0.0498753117,0.0748129676,0.0249376559,
                        0.0374064838,0.0623441397,0.0024937656,0.0124688279,0.2244389027]
    obsMod['Stratocumuliform'] = [0.0265957447,0.0531914894,0.0930851064,0.039893617,0.0664893617,
                                0.079787234,0.1861702128,0.079787234,0.0132978723,0.0265957447,
                                0.0531914894,0.0265957447,0.0026595745,0.0132978723,0.2393617021]
    obsMod['Cumuliform'] = [0.0269541779,0.0539083558,0.0673854447,0.0404312668,0.0808625337,
                          0.0539083558,0.0269541779,0.0539083558,0.0404312668,0.1886792453,
                          0.0943396226,0.0134770889,0.0026954178,0.0134770889,0.2425876011]
    obsMod['Cumulonibiform'] = [0.0020491803,0.006147541,0.1844262295,0.0020491803,0.006147541,
                              0.1844262295,0.0020491803,0.0040983607,0.1946721311,0.0102459016,
                              0.0040983607,0.1926229508,0.1844262295,0.0204918033,0.0020491803]
    alphas = {}
    for f in famNames:
        alphas[f] = [-1,-1]

    probs = {}
    for f in famNames:
        probs[f] = 1

    #for each bit of data
    for d in data:
        #update classification probs
        for f in famNames:
            alphas[f] = continueForward(d,models[f],alphas[f])
            probs[f] = probs[f]*sum(alphas[f])

        #normalize probs
        suma = sum(probs.values())
        for f in famNames:
            probs[f] = probs[f]/suma

        #show to human
        #get human observation
        #print(probs)
        #ob = int(raw_input("Which observation would you like to make?"))

        #apply bayes rule
        for f in famNames:
            probs[f] = probs[f]*obsMod[f][ob]

        #normalize probs
        suma = sum(probs.values())
        for f in famNames:
            probs[f] = probs[f]/suma

        self.probs = probs
        time.sleep(2)
        # #update alphas
        # for f in famNames:
        # 	for i in range(len(alphas[f])):
        # 		alphas[f][i] = alphas[f][i]*probs[f]
