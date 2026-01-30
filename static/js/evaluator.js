// Resume Evaluator specific JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const evaluationForm = document.getElementById('evaluationForm');
    const feedbackForm = document.getElementById('feedbackForm');
    const resultDiv = document.getElementById('evaluation-result');
    const submitBtn = document.getElementById('submitBtn');

    if (evaluationForm) {
        evaluationForm.addEventListener('submit', handleEvaluationSubmit);
    }

    if (feedbackForm) {
        feedbackForm.addEventListener('submit', handleFeedbackSubmit);
    }
});

async function handleEvaluationSubmit(e) {
    e.preventDefault();
    if (!utils.validateForm('evaluationForm')) {
        return;
    }

    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Evaluating...';

    // Show progress indicator
    const progressDiv = document.createElement('div');
    progressDiv.id = 'progress-indicator';
    progressDiv.innerHTML = `
        <div class="progress mb-3">
            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
        </div>
        <div id="progress-message" class="text-center text-muted">Starting analysis...</div>
    `;
    
    const resultDiv = document.getElementById('evaluation-result');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '';
    resultDiv.appendChild(progressDiv);

    try {
        const formData = new FormData(e.target);
        async function getResponseErrorMessage(response) {
            try {
                const ct = response.headers.get('content-type') || '';
                if (ct.includes('application/json')) {
                    const data = await response.json();
                    return data.error || data.message || JSON.stringify(data);
                }
            } catch (_) {}
            try {
                const txt = await response.text();
                if (txt) return txt;
            } catch (_) {}
            return `Request failed (${response.status})`;
        }
        
        // Use streaming endpoint
        const response = await fetch('/evaluate-stream', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const msg = await getResponseErrorMessage(response);
            throw new Error(msg);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let progressBar = document.querySelector('.progress-bar');
        let progressMessage = document.getElementById('progress-message');
        let currentStep = 0;
        const totalSteps = 4;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.status === 'processing') {
                            progressMessage.textContent = data.message;
                            progressBar.style.width = '10%';
                        } else if (data.status === 'step1') {
                            progressMessage.textContent = data.message;
                            progressBar.style.width = '25%';
                        } else if (data.status === 'basic_results') {
                            // Show basic results immediately (5-8 seconds instead of 20)
                            progressMessage.textContent = 'Basic analysis complete! Loading additional details...';
                            progressBar.style.width = '50%';
                            displayBasicResults(data);
                        } else if (data.status === 'step2') {
                            progressMessage.textContent = data.message;
                            progressBar.style.width = '60%';
                        } else if (data.status === 'additional_data') {
                            progressMessage.textContent = 'Loading job stability and career analysis...';
                            progressBar.style.width = '75%';
                            displayAdditionalData(data);
                        } else if (data.status === 'step3') {
                            progressMessage.textContent = data.message;
                            progressBar.style.width = '85%';
                        } else if (data.status === 'questions') {
                            progressMessage.textContent = 'Loading interview questions...';
                            progressBar.style.width = '95%';
                            displayQuestions(data);
                        } else if (data.status === 'step4') {
                            progressMessage.textContent = data.message;
                            progressBar.style.width = '98%';
                        } else if (data.status === 'complete') {
                            progressMessage.textContent = 'Analysis complete!';
                            progressBar.style.width = '100%';
                            progressBar.classList.remove('progress-bar-animated');
                            progressBar.classList.add('bg-success');
                            
                            // Hide progress indicator after 2 seconds
                            setTimeout(() => {
                                progressDiv.style.display = 'none';
                            }, 2000);
                        } else if (data.status === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (parseError) {
                        console.error('Error parsing SSE data:', parseError);
                    }
                }
            }
        }
    } catch (error) {
        utils.handleError(error);
        progressDiv.innerHTML = '<div class="alert alert-danger">Error: ' + error.message + '</div>';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Evaluate Resume';
    }
}

function displayBasicResults(data) {
    const resultDiv = document.getElementById('evaluation-result');
    
    // Update match score
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        utils.animateProgressBar(progressBar, data.match_percentage);
    }

    // Update match factors
    updateMatchFactors(data.match_factors);

    // Update profile information
    updateProfileInfo(data);

    // Set evaluation ID for feedback
    document.getElementById('evaluation-id').value = data.id;
}

function displayAdditionalData(data) {
    // Update job stability
    updateJobStability(data.job_stability);
    
    // Update career progression
    updateCareerProgression(data.career_progression);
}

function displayQuestions(data) {
    // Update interview questions
    updateInterviewQuestions(data);
}

function displayResults(data) {
    const resultDiv = document.getElementById('evaluation-result');
    resultDiv.style.display = 'block';

    // Update match score
    const progressBar = document.getElementById('progress-bar');
    utils.animateProgressBar(progressBar, data.match_percentage);

    // Update match factors
    updateMatchFactors(data.match_factors);

    // Update job stability
    updateJobStability(data.job_stability);

    // Update profile information
    updateProfileInfo(data);

    // Update interview questions
    updateInterviewQuestions(data);

    // Set evaluation ID for feedback
    document.getElementById('evaluation-id').value = data.id;
}

function updateMatchFactors(factors) {
    Object.entries(factors).forEach(([factor, value]) => {
        const element = document.getElementById(factor.toLowerCase().replace(' ', '-'));
        if (element) {
            element.style.width = `${value}%`;
            element.nextElementSibling.textContent = `${value}%`;
        }
    });
}

function updateJobStability(stability) {
    const stabilityBar = document.getElementById('stability-score-bar');
    utils.animateProgressBar(stabilityBar, stability.StabilityScore);

    document.getElementById('risk-level').textContent = stability.RiskLevel;
    document.getElementById('risk-level').className = `badge bg-${getRiskLevelColor(stability.RiskLevel)}`;
    document.getElementById('average-tenure').textContent = stability.AverageJobTenure;
    document.getElementById('job-count').textContent = stability.JobCount;
    document.getElementById('stability-explanation').textContent = stability.ReasoningExplanation;
}

function getRiskLevelColor(riskLevel) {
    switch (riskLevel.toLowerCase()) {
        case 'low': return 'success';
        case 'medium': return 'warning';
        case 'high': return 'danger';
        default: return 'secondary';
    }
}

function updateProfileInfo(data) {
    document.getElementById('profile-summary').textContent = data.profile_summary;
    
    const missingKeywords = document.getElementById('missing-keywords');
    missingKeywords.innerHTML = data.missing_keywords.length > 0 
        ? `<ul>${data.missing_keywords.map(kw => `<li>${kw}</li>`).join('')}</ul>` 
        : 'No missing keywords identified.';
    
    document.getElementById('extra-info').textContent = data.extra_info || 'No additional information provided.';
}

function updateInterviewQuestions(data) {
    const questions = {
        'quick-checks-questions': data.behavioral_questions,
        'soft-skills-questions': data.nontechnical_questions,
        'technical-skills-questions': data.technical_questions
    };

    Object.entries(questions).forEach(([listId, questionList]) => {
        const list = document.getElementById(listId);
        if (list) {
            list.innerHTML = questionList.map(q => `<li class="list-group-item">${q}</li>`).join('');
        }
    });
}

async function handleFeedbackSubmit(e) {
    e.preventDefault();
    if (!utils.validateForm('feedbackForm')) {
        return;
    }

    try {
        const formData = new FormData(e.target);
        const feedbackData = {
            evaluation_id: formData.get('evaluation_id'),
            rating: formData.get('rating'),
            comments: formData.get('comments')
        };

        await utils.submitFeedback(
            feedbackData.evaluation_id,
            feedbackData.rating,
            feedbackData.comments
        );

        alert('Feedback submitted successfully!');
        e.target.reset();
    } catch (error) {
        utils.handleError(error);
    }
} 