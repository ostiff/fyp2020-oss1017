const domain = document.location.origin;
const data_url = domain + "/get_data";
const KDTree_url = domain + "/get_k_nearest"
const encPatient_url = domain + "/enc_patient"

let N = 0;

async function getData(url) {
    let response = await fetch(url);
    return await response.json();
}

function updatePage(url) {
    let tableSummary = document.getElementById("tableSummary");
    getData(url).then((data) => {
        let colours = Array(N).fill('#1f77b4');
        for(let i=0; i < data.idx[0].length; i++){
            colours[data.idx[0][i]] = '#ff7f0e';
        }

        let update = {'marker':{color: colours}};
        Plotly.restyle('scatterPlot', update, 0);

        tableSummary.innerHTML = data.table;
    });
}

function encPatient() {
    let form = document.getElementById("encode-form");
    let query_url = new URL(encPatient_url);
    query_url.searchParams.append('k', $( "#selectK" ).slider( "option", "value" ));

    let params = ["age", "weight", "plt", "hct", "b_temp"];

    for (let i = 0; i < form.length - 1 ;i++) {
        query_url.searchParams.append(params[i], form.elements[i].value);
    }

    updatePage(query_url);
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
    N = scatter_data.x.length;

    let data = {
        x: scatter_data.x,
        y: scatter_data.y,
        mode: 'markers',
        type: 'scatter',
        marker: {
            color: '#1f77b4'
        }
    };

    let layout = {
        hovermode:'closest',
        title:'Auto-Encoder - Latent dimension',
        // yaxis: {
        //     automargin: true,
        //     scaleanchor: "x",
        // },
        margin: {
            t: 50,
            l: 50,
            r: 50,
            b: 50
        }
    };

    Plotly.newPlot('scatterPlot', [data], layout);
    window.addEventListener('resize', resizePlot);
    $(".sidebar-toggle").click(function(){
        resizePlot();
    });
    let scatterPlot = document.getElementById('scatterPlot');

    let k = $( "#selectK" ).slider( "option", "value" );

    $( "#selectK" ).slider({
        max: N-1
    });

    $('#listenSlider').change(function() {
        $('.output b').text( this.value );
        k = this.value;
    });


    scatterPlot.on('plotly_click', function(data){
        let id = data.points[0].pointIndex;
        let query_url = new URL(KDTree_url);
        query_url.searchParams.append('id', id);
        query_url.searchParams.append('k', k);

        updatePage(query_url);
    });

    // validation summary
    let $encodeForm = $("#encode-form");
    $encodeForm.validate({
        errorContainer: $encodeForm.find( 'div.validation-message' ),
        errorLabelContainer: $encodeForm.find( 'div.validation-message ul' ),
        wrapper: "li"
    });
}

main();
