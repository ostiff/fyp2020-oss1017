const domain = document.location.origin;
const data_url = domain + "/get_data";
const trace_url = domain + "/get_trace";
let trace_data = {};

async function getData(url) {
    let response = await fetch(url);
    return await response.json();
}

function updateColour(day, N) {
    let colours = [];
    for(let i=0; i < day; i++){
        colours.push('#ff7f0e');
    }
    for(let i=day; i < N; i++){
        colours.push('#000000');
    }
    let update = {'marker':{color: colours, size: 15}};
    Plotly.restyle('scatterPlot', update, 1);
}

async function updateTrace(val){
    let query_url = new URL(trace_url);
    if(val) {
        query_url.searchParams.append('study_no', val);
    }
    trace_data = await getData(query_url);

    let update = {
        'x': [trace_data.x],
        'y': [trace_data.y],
        name: trace_data.study_no
    };
    $( "#dateSlider" ).slider({
        max: trace_data.x.length,
        value: 1
    });
    $('#dispDate b').text( trace_data.date[0] );

    await Plotly.restyle('scatterPlot', update, 1);
    updateColour(1, trace_data.x.length);
}

function submitForm() {
    let form = document.getElementById("studynoForm");
    updateTrace(form.elements[0].value)
}

function resizePlot() {
    let width = $("#scatterPlot").parent().width();
    let height = $("#scatterPlot").parent().height();
    Plotly.relayout('scatterPlot', {
        width: width,
        height: height
    })
}

async function main() {

    const scatter_data = await getData(data_url)

    let general_trace = {
        x: scatter_data.x,
        y: scatter_data.y,
        mode: 'markers',
        type: 'scatter',
        marker: {
            color: '#1f77b4'
        },
        name: 'All data'
    };

    let patient_trace = {
        x: [],
        y: [],
        mode: 'lines+markers',
        type: 'scatter',
        marker: {
            size: 15,
            color: '#000000'
        },
        line: {
            color: '#000000'
        }
    };

    let layout = {
        hovermode:'closest',
        title:'Auto-Encoder - Latent dimension',
        margin: {
            t: 50,
            l: 50,
            r: 50,
            b: 50
        }
    };

    Plotly.newPlot('scatterPlot', [general_trace, patient_trace], layout);
    window.addEventListener('resize', resizePlot);
    $(".sidebar-toggle").click(function(){
        resizePlot();
    });

    $('#listenSlider').change(function() {
        updateColour(this.value, this.max);
        $('#dispDate b').text( trace_data.date[this.value-1] );
    });
}

main();
