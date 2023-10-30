import math # pow(), e

def sigmoid(x):
    # return 1 / ( 1 + e^-x )
    return 1 / ( 1 + math.pow( math.e, (-x) ) )

def compute_neural_network(inputs, weights, biases):
    # inputs is a list of normals
    # wieghts is a 3-dimensaional list floats
    # biases is a 2-dimensional list of floats
    output = inputs
    for layerW, layerB in zip( weights, biases ):
        layerOut = []
        # 1) Dot multiply layerInput with network weights:
        for w in layerW:
            layerOut.append( [ o * wi for o, wi in zip( output, w ) ] )
        # 1.1) Summation step of dot multiplication:
        layerOut = [ sum(o) for o in layerOut ]
        # 2) Add layer biases:
        layerOut = [ o + b for o, b in zip( layerOut, layerB ) ]
        # 3) Sigmoid function:
        layerOut = [ sigmoid(x) for x in layerOut ]
        output = layerOut
    return output


if __name__ == "__main__":
    ex_inputs = [[0.0], [0.1], [0.49], [0.5], [0.51], [0.9], [1.0]]
    def demo( weights, biases, test_inputs=ex_inputs ):
        for t in test_inputs:
            print( t, compute_neural_network( t, weights, biases ) )
        print("")
    
    print("Compression:")
    #  [[[Compression]]], [[-Compression/2]]
    demo([[[1000]]], [[-500]])
    
    print("Inverting Compression:")
    #  [[[-Compression]]], [[Compression/2]]
    demo([[[-1000]]], [[500]])
    
    print( "Output Hypotheses: [ InputIsHigh, InputIsLow ]" )
    # [[[Compression],[-Compression]]], [[Compression/2]]
    demo([[[1000],[-1000]]], [[-500, 500]])

  
    ex_input_pairs = [[0.0,0.0], [0.0,1.0], \
        [1.0,0.0], [1.0,1.0]]
  
    print("Logical \"Or\":")
    # [[[Compression,Compression]]], [[Compression/2]]
    demo([[[1000,1000]]], [[-500]], ex_input_pairs)
    
    print("Logical \"And\":")
    # [[[Compression/2,Compression/2]]], [[(Compression/2)+1]]
    demo([[[500,500]]], [[-501]], ex_input_pairs)
    
    print("No \"Not\"s permitted. Instead use a negative compression layer first.")
    demo([[[1000,0],[0,-1000]],[[500,500]]], [[-500,500],[-501]], ex_input_pairs)
