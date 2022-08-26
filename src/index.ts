const ort = require('onnxruntime-node')
import schema from '../schema.json'

const BIGINTKEY = "int64"
interface Dictionary<T> {
    [Key: string]: T;
}

//this converts [{key:value}] into {key:[value]}
export const transposeRows = (schema: Dictionary<any>, arr: Dictionary<any>[], keys: string[]) => {
    return arr.reduce((aggr, row) => {
        for (const key of keys) {
            if (schema[key] === BIGINTKEY) {
                aggr[key].push(BigInt(row[key]))
            }
            else {
                aggr[key].push(row[key])
            }

        }
        return aggr
    }, keys.reduce((aggr, key) => ({
        ...aggr,
        [key]: []
    }), {}))
}
export const parseSchema = (schema: Dictionary<any>, arr: Dictionary<any>[]) => {
    const keys = Object.keys(schema)
    const transposedRows = transposeRows(schema, arr, keys)
    const n = arr.length
    return keys.reduce((aggr, key) => {
        return { ...aggr, [key]: new ort.Tensor(schema[key], transposedRows[key], [n, 1]) }
    }, {})
}

interface LabelOutput {
    dims: number[],
    type: string,
    size: number,
    data: any[]
}
interface ProbabilityOutput {
    dims: number[],
    type: string,
    size: number,
    data: Float32Array
}
interface ModelOutput {
    label: LabelOutput,
    probabilities: ProbabilityOutput
}
interface Predictions {
    value: any,
    probability: number
}

const getMaxNumber = (arr: Float32Array) => {
    let maxElem = Number.MIN_VALUE
    for (const v of arr) {
        if (v > maxElem) {
            maxElem = v
        }
    }
    return maxElem
}
export const parseOutput = (results: ModelOutput) => {
    const numLabels = results.probabilities.dims[1]
    const parsedResults: Predictions[] = []
    console.log(results)
    results.label.data.forEach((v, index) => {
        const value = results.label.type === BIGINTKEY ? parseInt(v.toString()) : v
        const probability = getMaxNumber(results.probabilities.data.slice(index * numLabels, (index + 1) * numLabels))
        const result = {
            value, probability
        }
        parsedResults.push(result)
    })
    return parsedResults
}


exports.handler = async function (event: any, context: any) {
    //TODO, make this an environment variable OR force the same model name during docker build
    const session = await ort.InferenceSession.create('./pipeline_titanic.onnx');
    const feeds = parseSchema(schema, event.body)
    const results = await session.run(feeds);
    return parseOutput(results)
}

// use an async context to call onnxruntime functions.
async function main() {
    try {
        const session = await ort.InferenceSession.create('./pipeline_titanic.onnx');

        const feeds = parseSchema(schema, [{
            sex: "female",
            age: 20.0,
            fare: 3.5,
            embarked: "S",
            pclass: 1
        },
        {
            sex: "male",
            age: 25.0,
            fare: 6.0,
            embarked: "S",
            pclass: 2
        },
        {
            sex: "male",
            age: 25.0,
            fare: 6.0,
            embarked: "S",
            pclass: 2
        }])

        const results = await session.run(feeds);
        console.log(parseOutput(results))

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

//main();